from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
from json.decoder import JSONDecodeError
import re






#---------------------------------------------
# HuggingFace Login & LLM Setup
#---------------------------------------------

# HuggingFace Login
access_token = "paste_your_token_here"
login(token=access_token)

# load the tokenizer and the model
model_name = "Qwen/Qwen3-30B-A3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)




#---------------------------------------------
# Reading the input .json file
#---------------------------------------------

def load_json(file_path):
    """
    functiont to parse the .json file and extract the input dictionaries
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

input_file = "Input.json"
input_data = load_json(input_file)




#---------------------------------------------
# LLM Prompt (includes Few-shot examples)
#---------------------------------------------

# prepare the model input
prompt = """
### BACKGROUND ###
You are an assistant specialized in Air Traffic Control (ATC) communications between an air traffic controller and a pilot. You know the phraseology used in ATC communications and understand the importance of each word in the communication. The reference is the correct version of the communication between the pilot and the controller and, the transcription is a version of the reference which may or may not contain errors such as spelling mistakes, word omissions, word additions, word substitutions or missing word/words.


### INSTRUCTION ###
You must compare the reference and transcript given in the query and generate ONLY the following-
    1. analysis: A detailed explanation of the errors identified in the transcript when compared with the reference statement and whether they substantially change the core message conveyed in the communication
    2. errors_identified: A list of strings of the type of errors described in the analysis. Example- ["Key information error", "Callsign error", "Minor error"]. If there minor errors no errors identified the list will be empty, Example- []


### DEFINITIONS ###
    1. Callsign: A callsign is a unique identifier used in aviation to communicate the identity of an aircraft during radio communications with air traffic control (ATC). It is used to ensure clarity and precision when referring to a specific aircraft, especially during flight operations. Callsigns can vary based on the airline, flight, region, and specific operational purpose. Below is a detailed breakdown of its components and the importance of a callsign:
        
        a) Components of a Callsign: A callsign typically consists of two primary components:
           1) Airline Designator: This is a two- or three-letter code representing the airline operating the flight. It is assigned by the International Civil Aviation Organization (ICAO) or the airline itself
             - Examples: ('american', 'delta', 'united', 'southwest', 'jetblue', 'alaska', 'spirit', 'lufthansa', 'air france', 'emirates', 'qatar', etc.)
           2) Flight Number: This is a numerical identifier for a specific flight operated by the airline. It is typically 1 to 4 digits long, though variations exist depending on the airline and region. The flight number can be followed by a suffix (e.g.,  "Alpha", "Bravo") in cases where there are multiple flights with the same number.
             - Examples: ('one seven nine', 'four six eight heavy', 'zero zero nine four', 'one five bravo')
             
        b) Therefore, examples of some callsigns are:
           1) 'soutwest seven seven two'
           2) 'delta five one zero'
           3) 'spirit four fifty'
           4) 'united one two zero heavy'
            
        c) Importance of a callsign:
           Callsigns help ensure clear, unambiguous communication between pilots and ATC.
           
       **Hence, when comparing a transcript with a reference statement, if airline designator or the flight number in the transcript have an error, it considered a 'Callsign error' as it results in substantial loss of information, context and meaning etc.**
           
    2. Key information: In air traffic communications, the controller and pilot communicate about several important details that ensure smooth and safe airline operations:
        a) Cleanrances; The controller provides take-off/landing, climb/descend, approach, runway and other crucial clearances to the pilot
        b) Instuctions; The controller provides navigation related instruction such as heading change, altitude change, speed change and other emergency navigation instructions to the pilot
        c) Important information; The controller also provides other important information such as tower contact details, traffic alerts etc
        d) Acknowledgement; The pilot usually acknowledges the instructions and clearances given by the controller and also conveys any emergency information
        
       **Hence, when comparing a transcript with a reference statement, if any of the above 4 types information in the transcript have an error, it considered a 'Key information error' as it results in substantial loss of information, context and meaning etc.**
        
    3. Minor information: In air traffic communications, the following is considered minor information and are not important to the meaning of the communication
        a) Greetings shared by the pilots and the controlers
        b) Words like <uh>, <speech>, <noise> which indicate mumbling or unclear speech
        c) Any other piece of communication that does not hold any contextual importance to the core meaning of the communication

       **Hence, when comparing a transcript with a reference statement, if any of the above mentioned 3 types of information in the transcript have an error, it is considered a 'minor error' as does NOT result in substantial loss of information, context and meaning etc.**
        

### EXAMPLES ###
Given below are some examples where the transcript has some errors (when comapared word-for-word with the reference). Here, 'analysis' **CORRECTLY** explains the significance of those errors in the context to safety of airline operations and 'errors_identified' **CORRECTLY** indicates the errors that were described in the analysis.
Correct example 1:
    reference: "swiss five two heavy turn right heading of <uh> zero three zero intercept the localizer"
    transcript: "swiss five turn right heading of zero three zero intercept the localizer"
    analysis: "On comparing the transcript with the reference, the word '<uh>' is omitted in the transcript however, it does not important to core meaning of the communication. The omission of the phrase 'two heavy' in the transcript has introduced an error in the aircraft callsign as a part of it is missing"
    errors_identified: ["Callsign error"]

Correct example 2:
    reference: "piedmont forty eight seventy two descend and maintain three thousand"
    transcript: "piedmont forty eight seventy two ascend maintain three thousand"
    analysis: "On comparing the transcript with the reference, the substitution of the phrase 'descend and' with 'ascend' in the transcript changes the main intent of the communication which was to instruct the pilot to descend"
    errors_identified: ["Key information error"]

Correct example 3:
    reference: "jetblue ten eighty four turn left heading of two five zero intercept the localizer"
    transcript: "jetglue ten eighty four turn left heading of five zero intercept the localizer"
    analysis: "On comparing the transcript with the reference, the substitution of the word 'jetblue' with 'jetglue' in the transcript has introduced an error in the aircraft identifier. The omission of the word 'two' has introduced an error in the heading information conveyed by the air traffic controller"
    errors_identified: ["Callsign error", "Key information error"]

Correct example 4:
    reference: "twenty five twenty eight going to departure <noise> good day now"
    transcript: "twenty five twenty eight going to departure"
    analysis: "On comparing the transcript with the reference, the omission of the phrase 'good day now' in the transcript causes no loss of critical information"
    errors identified: ["Minor error"]

Given below are some examples where the transcript has some errors (when comapared word-for-word with the reference). However, here both 'analysis' and 'errors_identified' are **INCORRECT**
Incorrect example 1:
    reference: "execjet six zero one roger <uh> turn left heading one five zero"
    transcript: "execjet six zero one roger <uh> turn heading one five zero"
    analysis: "On comparing the transcript with the reference, the omission of the word 'left' in the transcript causes no loss of critical information as the core meaning of the communication to turn is still conveyed"
    errors identified: ["Minor error"]

Incorrect example 2:
    reference: "shamrock one alfa charlie descend and maintain three thousand"
    transcript: "shamrock one on alfa charlie descend and three thousand"
    analysis: "Addition of the word 'on' in the transcript has introduced a minor error in the aircraft identifier. The omission of the word 'maintain' in the transcript was not crucial for the communication"
    errors identified: ["Minor error"]

Incorrect example 3:
    reference: "abex one ten heavy on departure fly heading two eight five runway two six left cleared for takeoff"
    transcript: "abex one ten heavy on departure fly heading two eight five runway two six left cleared takeoff"
    analysis: "On comparing the transcript with the reference, the substitution of the word 'abex with 'abex' has introduced an error in the aircraft identifier which is a cricial part of the communication"
    errors identified: ["Key information error"]

Incorrect example 4:
    reference: "air berlin seventy four seventy two heavy traffic completion of turn ten o'clock seven miles northeast bound is an embraer inbound for the parallel runway they'll have you in sight maintaining visual separation with you"
    transcript: "air berlin seventy four seventy two heavy traffic completion of turn ten o'clock seven miles east bound is an embraer inbound for the parallel runway they'll have you in sight maintaining visual separation with you"
    analysis: "On comparing the transcript with the reference, the substitution of the word 'northeast' with 'east' in the transcript has causes no loss of critical information as the direction information is still conveyed.
    errors identified: ["Minor error"]


### OUTPUT FORMAT ###
The output should be formatted as a JSON instance that conforms to the JSON schema below.
```
{{
'analysis':
'errors_identified':
}}
```


### QUERY ###
reference: {reference}
transcript: {transcript}


### CORRECT ANSWER ###
"""




#---------------------------------------------
# Generating & saving the output
#---------------------------------------------

def generate_response(model, tokenizer, final_prompt):
    """
    xxx
    """
    messages = [{"role": "user", "content": final_prompt}]
    text = tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)       #Default is True.
                                         
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs,
                                   max_new_tokens=30000)
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thought_process = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return thought_process, response


storage = []

for input_query in input_data:
    
    # if the transcript is a perfectly matches the reference, an LLM analysis is not required
    if input_query["reference"] == input_query["transcription"]:
        response = {
                    "transcript": input_query["transcription"],
                    "reference": input_query["reference"],
                    "analysis": "No errors found",
                    "errors_identified": []
                    }
                    
        storage.append(json_obj)
        continue
        
    final_prompt = prompt.format(reference=input_query["reference"],
                                 transcript=input_query["transcription"])
    
    # generating the response for the reference-transcript pair
    thought_process, response = generate_response(model, tokenizer, final_prompt)
    
    # processing the raw response
    match = re.search(r"\{\s*[\s\S]*?\}", response)  # Matches a JSON-like structure inside {}
    
    if match:
        try:
            json_str = match.group()  # Extract matched JSON substring
            json_obj = json.loads(json_str)  # Convert to Python dictionary
            processed_response = {
                                  "transcript": input_query["transcription"],
                                  "reference": input_query["reference"],
                                  "analysis": json_obj["analysis"],
                                  "errors_identified": json_obj["errors_identified"]
                                  }
            storage.append(processed_response)

        except JSONDecodeError:
            processed_response = {
                                  "transcript": input_query["transcription"],
                                  "reference": input_query["reference"],
                                  "ERROR_MESSAGE": "Computed the response to this query but could not extract json obj",
                                  "LLM_response": response
                                  }            
            storage.append(processed_response)

    else:
        processed_response = {
                              "transcript": input_query["transcription"],
                              "reference": input_query["reference"],
                              "ERROR_MESSAGE": "Computed the response to this query but no json obj found",
                              "LLM_response": response
                              }          
        storage.append(processed_response)
    
    print(f"Finished ananlyzing sample {input_data.index(input_query)}")
    
    
# Saving the output
with open("Output.json", "w") as f:
    json.dump(storage, f, indent=4)
