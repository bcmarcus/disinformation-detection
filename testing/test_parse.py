import re

def parse_output(output):
    # Match the model's response part only, ignoring the prompt part
    model_response_pattern = re.compile(r"<start_of_turn>model\s*(.*)", re.DOTALL)
    model_response_match = model_response_pattern.search(output)
    
    if not model_response_match:
        return "No answer found", "No explanation found"
    
    model_response = model_response_match.group(1).strip()

    # Now extract the answer and explanation from the model's response
    answer_pattern = re.compile(r"Answer:\s*(.*?)\s*Explanation:", re.DOTALL)
    explanation_pattern = re.compile(r"Explanation:\s*(.*)", re.DOTALL)

    answer_match = answer_pattern.search(model_response)
    explanation_match = explanation_pattern.search(model_response)
    
    answer = answer_match.group(1).strip() if answer_match else "No answer found"
    explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found"

    return answer, explanation

output = """
<bos><start_of_turn>user
    Use the following pieces of context to determine the veracity of the statement and explain why in a concise manner (1-2 sentences max).
    Use the format Answer: *the actual answer* Explanation: *the actual explanation*.

    Context:
    Despite record numbers of COVID-19 cases in the U.S. from mid-June onward and an increasing percentage of positive test results, Trump largely continued to downplay the pandemic, including his false claim in early July 2020 that 99 percent of COVID-19 cases are "totally harmless". He began insisting that all states should resume in-person education in the fall despite a July spike in reported

On October 2, 2020, Trump tweeted that he had tested positive for COVID-19, part of a White House outbreak. Later that day Trump was hospitalized at Walter Reed National Military Medical Center, reportedly due to fever and labored breathing. He was treated with antiviral and experimental antibody drugs and a steroid. He returned to the White House on October 5, still infectious and unwell.

By July 2020, Trump's handling of the COVID-19 pandemic had become a major issue in the presidential election. Biden sought to make the pandemic the central issue. Polls suggested voters blamed Trump for his pandemic response and disbelieved his rhetoric concerning the virus, with an Ipsos/ABC News poll indicating 65 percent of respondents disapproved of his pandemic response. In the final

and it can be considered in those who have been diagnosed with COVID‑19 and have an altered mental status.

    Statement: Trump doesnt believe in Covid-19.<end_of_turn>
    <start_of_turn>model
    Answer: The statement is false. 
    Explanation: The provided context shows Trump downplaying the pandemic, making false claims, and even contracting COVID-19 himself. 
"""

parsed_answer, parsed_explanation = parse_output(output)
print("Parsed Answer:", parsed_answer)
print("Parsed Explanation:", parsed_explanation)
