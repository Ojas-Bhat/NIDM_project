import pandas as pd

rows = [
    {"question":"What is the definition of a cyclone?",
     "expected_answer":"A cyclone is a large-scale air mass rotating around a strong center of low pressure.",
     "true_disaster":"Cyclone"},
    {"question":"What causes cyclones to form?",
     "expected_answer":"Cyclones form due to warm ocean waters, moist air, and the Coriolis force.",
     "true_disaster":"Cyclone"},
    {"question":"What are the main categories of cyclones in India?",
     "expected_answer":"Depression; Deep Depression; Cyclonic Storm; Severe Cyclonic Storm; Very Severe Cyclonic Storm; Extremely Severe Cyclonic Storm; Super Cyclonic Storm.",
     "true_disaster":"Cyclone"},
    {"question":"What is the minimum wind speed for a cyclonic storm?",
     "expected_answer":"Wind speed must reach 63 kmph.",
     "true_disaster":"Cyclone"},
    {"question":"What wind speed defines a Very Severe Cyclonic Storm?",
     "expected_answer":"Wind speeds of 118 to 165 kmph.",
     "true_disaster":"Cyclone"},
    {"question":"In which region do most Indian cyclones originate?",
     "expected_answer":"The Bay of Bengal region.",
     "true_disaster":"Cyclone"},
    {"question":"Which coast of India is more vulnerable to cyclones?",
     "expected_answer":"The East Coast of India.",
     "true_disaster":"Cyclone"},
    {"question":"Name two states frequently affected by cyclones.",
     "expected_answer":"Odisha and Andhra Pradesh.",
     "true_disaster":"Cyclone"},
    {"question":"What is storm surge?",
     "expected_answer":"Storm surge is an abnormal rise in sea level caused by a cyclone.",
     "true_disaster":"Cyclone"},
    {"question":"Name one major impact of storm surge.",
     "expected_answer":"Coastal flooding.",
     "true_disaster":"Cyclone"},
    {"question":"What is the main hazard associated with cyclone winds?",
     "expected_answer":"Destruction of buildings and infrastructure.",
     "true_disaster":"Cyclone"},
    {"question":"What are cyclone preparedness measures?",
     "expected_answer":"Evacuation, early warnings, shelters, and coastal embankments.",
     "true_disaster":"Cyclone"},
    {"question":"What is the role of IMD in cyclones?",
     "expected_answer":"IMD monitors, predicts, and issues cyclone warnings.",
     "true_disaster":"Cyclone"},
    {"question":"What is cyclone mitigation?",
     "expected_answer":"Actions taken to reduce long-term cyclone risks.",
     "true_disaster":"Cyclone"},
    {"question":"Which season sees most cyclones in India?",
     "expected_answer":"Pre-monsoon and post-monsoon seasons.",
     "true_disaster":"Cyclone"},
    {"question":"What should communities do before a cyclone?",
     "expected_answer":"Secure homes, store supplies, and follow evacuation warnings.",
     "true_disaster":"Cyclone"},
    {"question":"What should people avoid during a cyclone?",
     "expected_answer":"Avoid going outdoors or near the coastline.",
     "true_disaster":"Cyclone"},
    {"question":"What is a cyclone shelter?",
     "expected_answer":"A safe building designed to protect people during cyclones.",
     "true_disaster":"Cyclone"},
    {"question":"Why is the Bay of Bengal more prone to cyclones?",
     "expected_answer":"Because of warmer waters and favorable wind conditions.",
     "true_disaster":"Cyclone"},
    {"question":"What is one post-cyclone recovery measure?",
     "expected_answer":"Restoring power, roads, and essential services.",
     "true_disaster":"Cyclone"}
]

df = pd.DataFrame(rows)
df.to_csv("eval_questions1.csv", index=False, encoding="utf-8", quoting=1)  # quoting=1 => csv.QUOTE_ALL
print("Wrote eval_questions1.csv with", len(df), "rows")


import pandas as pd
df = pd.read_csv("eval_questions1.csv", encoding="utf-8")
print(df.head())
