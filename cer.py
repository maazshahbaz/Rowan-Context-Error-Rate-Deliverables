import json
import re
import torch

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# ------------------------------------------------------------------------
# 1. Hugging Face Model + Pipeline Setup
# ------------------------------------------------------------------------

# Login to Hugging Face (required if using gated models)
login(token="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")

# Define the model name — Meta LLaMA-3 70B Instruct
model_name = "meta-llama/Llama-3-70B-Instruct"

print("Loading tokenizer and model. This may take some time...")

# Configure 8-bit quantization for memory efficiency
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    bnb_8bit_compute_dtype=torch.float16
)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# Create HuggingFace pipeline for text generation
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.2,
    return_full_text=False
)

# Wrap pipeline in LangChain-compatible LLM object
llm = HuggingFacePipeline(pipeline=hf_pipeline)

print("Model and pipeline loaded successfully!")

# ------------------------------------------------------------------------
# 2. Prompt Template and Few-Shot Examples
# ------------------------------------------------------------------------

few_shot_examples = """
Example 1:
- Reference transcription: "american two zero three climb and maintain eight thousand"
- Transcribed audio: "american two zero three climb and maintain eight thousand"
Output:
1. Contextual Status: Contextually Equivalent
2. Errors Detected: None
3. Explanation: Both transcriptions are identical in operational meaning.

Example 2:
- Reference transcription: "delta five five six descend to four thousand and maintain visual separation"
- Transcribed audio: "delta five five six descend to five thousand and maintain visual separation"
Output:
1. Contextual Status: Contextually Different
2. Errors Detected: Numeric value changed (from four thousand to five thousand)
3. Explanation: The altitude difference may affect the intended operational instructions.

Example 3 (Minor Missing Word):
- Reference transcription: "delta two two four climb and maintain flight level three three zero"
- Transcribed audio: "delta two two four climb maintain flight level three three zero"
Output:
1. Contextual Status: Contextually Equivalent
2. Errors Detected: None
3. Explanation: The missing word 'and' does not alter the operational instruction.

Example 4:
- Reference transcription: "Right turn heading one niner zero; and [uh] four thousand till; what was the fix again?"
- Transcribed audio: "Right turn heading one niner zero; and [uh] four thousand till; what was the bricks again?"
Output:
1. Contextual Status: Contextually Different
2. Errors Detected: The phrase “fix” is replaced by “bricks” which obscures the question about the missing fix.
3. Explanation: It is no longer clear what the question is about; the critical “fix” information is lost.
"""

# Prompt template used by the model to generate structured responses
template = """
You are an aviation communication expert trained in Air Traffic Control (ATC) phraseology.

Analyze the following pair of transcriptions for contextual equivalence and operational accuracy:

- Reference transcription: {reference}
- Transcribed audio: {transcription}

### Instructions:
1. Compare the two transcriptions closely.
2. Focus on operational meanings, including:
   - Numeric values (e.g., altitudes, headings, frequencies).
   - Critical instructions (e.g., "climb to," "descend to," "line up and wait").
   - Locations, navigation points, and callsigns.
3. Identify errors that change the operational meaning:
   - Missing, incorrect, or additional numeric values.
   - Misidentified callsigns or navigation points.
   - Missing or altered instructions.
4. **Note**: If a word is missing (e.g., a filler or connector) but does **not** change the operational meaning, then the transcription is still considered "Contextually Equivalent."

Respond ONLY in this format:
1. Contextual Status: "Contextually Equivalent" or "Contextually Different"
2. Errors Detected: If none, write "None"; otherwise, list them clearly.
3. Explanation: Briefly explain how/why these errors matter.

Below are examples of how to evaluate the transcriptions:
""" + few_shot_examples

# Compile the template into a LangChain object
prompt_template = PromptTemplate(
    input_variables=["reference", "transcription"],
    template=template
)

# Create LangChain LLM chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# ------------------------------------------------------------------------
# 3. LLM Output Parser
# ------------------------------------------------------------------------

def parse_llm_output(llm_response: str):
    """
    Parse the structured output from the LLM into dictionary format.

    Parameters:
        llm_response (str): Raw text response from LLM

    Returns:
        dict: Parsed result with keys:
              - "Contextual Status"
              - "Errors Detected"
              - "Explanation"
    """
    result = {
        "Contextual Status": "",
        "Errors Detected": "",
        "Explanation": ""
    }

    lines = llm_response.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("1."):
            result["Contextual Status"] = line.split(":", 1)[-1].strip()
        elif line.startswith("2."):
            result["Errors Detected"] = line.split(":", 1)[-1].strip()
        elif line.startswith("3."):
            result["Explanation"] = line.split(":", 1)[-1].strip()

    return result

# ------------------------------------------------------------------------
# 4. Main Evaluation Logic
# ------------------------------------------------------------------------

def main(input_json_path, output_json_path):
    """
    Evaluate a batch of transcriptions and write the result to output JSON.

    Parameters:
        input_json_path (str): Path to input file (JSON with reference/transcription pairs)
        output_json_path (str): Path to save annotated results
    """
    with open(input_json_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    for item in data:
        reference_text = item["reference"]
        transcribed_text = item["transcription"]

        # Run LLM to evaluate contextual accuracy
        llm_response = chain.run(reference=reference_text, transcription=transcribed_text)

        # Parse model output and update item
        parsed_response = parse_llm_output(llm_response)
        item["Contextual Status"] = parsed_response["Contextual Status"]
        item["Errors Detected"] = parsed_response["Errors Detected"]
        item["Explanation"] = parsed_response["Explanation"]

    # Write final output
    with open(output_json_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------------
# 5. Execute Script
# ------------------------------------------------------------------------

if __name__ == "__main__":
    input_json_path = "/home/ashfaq93/SST/CER/Transcriptions/whisper-medium.en.json"
    output_json_path = "/home/ashfaq93/SST/CER/CER_Results/whisper-medium.en.json"
    
    main(input_json_path, output_json_path)
    print(f"✅ Done! Results saved to {output_json_path}")
