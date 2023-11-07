# Data Preparation for Fake News Detection in finetuning an LLM

We plan to take an example to explain our target data format.

## Pre-defined news class
In the two collected datasets, there are many types of news, such as 'True', 'False', 'pants-fire', 'hale-true', 'mostly-true', 'barely-true', 'half', and so on. For simplicity, we plan to unify the label types in the two datasets.

The mapping function is as follows:

* 'True' or 'true' is unified as 'True'.
* 'pants-fire', 'half-true', 'mostly-true', 'barely-true', 'half', 'false', 'False' are all unified as 'False'. That means "If the label is not true, then it is assigned the value 'False'."

## Raw Data Format
- "event_id": "8340.json"
- "claim": "A strong bipartisan majority in the House of Representatives voted to defund Obamacare."
- "label": "false"
- "explain": "Cruz said that \"a strong bipartisan majority\" in the House of Representatives \"voted to defund Obamacare.\" Even if you consider the overall 230-189 margin to be a \"strong\" victory for backers of the measure, it doesn’t qualify as \"bipartisan\" except in the most hyper-technical sense. In our book, that doesn’t qualify as much of a bipartisan action."

## Target Data for LLM Training
Every piece of data should be:

{

    "prompt": "Below is an instruction that describes a fake news detection task. Write a response that appropriately completes the request.\n\n### Instruction:\n If there are only True and False categories, based on your knowledge and the following information: Cruz said that \"a strong bipartisan majority\" in the House of Representatives \"voted to defund Obamacare.\" Even if you consider the overall 230-189 margin to be a \"strong\" victory for backers of the measure, it doesn’t qualify as \"bipartisan\" except in the most hyper-technical sense. In our book, that doesn’t qualify as much of a bipartisan action. Evaluate the following assertion: A strong bipartisan majority in the House of Representatives voted to defund Obamacare.' If possible, please also give the reasons. \n\n### Response:.", 
    
    "chosen": "According to our knowledge and the given information, we think that the claim is False.", 
    
    "rejected": "I don't know."

}

## Subsequent Google Search Engine and ChatGPT API Utilize
We consider enhancing the practicability of our proposed LLM for Fake News Detection via introducing google search engine and chatGPT.

The concrete procedure is shown as follows:
- Assume the given claim is :`A strong bipartisan majority in the House of Representatives voted to defund Obamacare.`
- Input the given claim to the API of ChatGPT to obtain the results of key works. That is `Pleas give the key words of the given claim: A strong bipartisan majority in the House of Representatives voted to defund Obamacare.` Note that the key words could be **Strong bipartisan majority** + **House of Representatives** + **Voted** + **Defund** + **Obamacare**
- Input the key words to Google Search Engine to get the search results. We only consider the text shown in the search pages, such as '2013年9月20日 — Republicans said the House vote showed bipartisan support for defunding Obamacare because two Democrats backed the GOP resolution', where we just select 'Republicans said the House vote showed bipartisan support for defunding Obamacare because two Democrats backed the GOP resolution' as the potential analysis material, like the `Explain`.
- Finally, the input of our LLM is `
If there are only True and False categories, based on your knowledge and the following information: 1. Republicans said the House vote showed bipartisan support for defunding Obamacare because two Democrats backed the GOP resolution'; 2. Republicans who support the bill say Obamacare was collapsing under its own weight as insurance companies stopped providing plans in some states ...; 3. U.S. Senate Republican leader Mitch McConnell announced a vote on a straight repeal of Obamacare, which would take effect in two years. Evaluate the following assertion: A strong bipartisan majority in the House of Representatives voted to defund Obamacare.' If possible, please also give the reasons. \n\n### Response:.` 
- Obatin the results from our proposed LLM for Fake News Detection.
- The End.
