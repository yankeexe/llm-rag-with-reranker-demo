"""Helper module demonstrating CSV cache data ingestion.

This module provides a simple example of how to create and write data to a CSV file
for caching purposes. It demonstrates basic CSV file handling using Python's csv module
to store question-answer pairs.
"""

import csv

data = [
    {
        "question": "What is the target candidate description",
        "answer": r"The target candidate is     expected to have a specific set of skills and experience in the field of Artificial Intelligence (AI) and Machine Learning (ML), with a focus on their use of AI/ML technologies within Amazon Web Services (AWS). The key characteristics of the target candidate include:\n\n- **Exposure to AI/ML technologies**: The candidate has up to 6 months of exposure to AI/ML technologies on AWS, but may not necessarily be building AI/ML solutions from scratch.\n- **Responsibility vs. Building**: The candidate uses AI/ML technologies in their work, but may not always be responsible for building or implementing these solutions.\n- **AWS Familiarity**: The candidate has a good understanding of the core AWS services (e.g., Amazon EC2, Amazon S3, AWS Lambda, and Amazon SageMaker) and their respective use cases.\n\nOverall, the target candidate is expected to demonstrate a solid foundation in AI/ML concepts and their application within AWS, while also being able to apply this knowledge to real-world scenarios.",
    },
    {
        "question": "What is AWS Certified AI Practitioner",
        "answer": r"AWS Certified AI Practitioner (AIF-C01) Exam Overview\n\nThe AWS Certified AI Practitioner (AIF-C01) exam is a professional certification program designed for individuals who can demonstrate overall knowledge of Artificial Intelligence (AI), Machine Learning (ML), and generative AI technologies, as well as their associated AWS services and tools. The exam aims to validate the candidate's ability to apply these concepts and technologies in real-world scenarios.\n\n## Key Objectives\n\nThe AWS Certified AI Practitioner exam objectives include:\n\n- **Understanding AI/ML Concepts**: The candidate should be able to demonstrate a broad understanding of AI, ML, and generative AI concepts, methods, and strategies.\n- **Applying AI/ML Technologies**: The candidate should be able to apply AI/ML and generative AI technologies in relevant ways within their organization.\n\n## Exam Designation\n\nThe AWS Certified AI Practitioner exam has a pass or fail designation. The exam is scored against a minimum standard established by AWS professionals, which ensures that candidates meet the required knowledge and skills for the certification.\n\n## What is Covered on the Exam?\n\nThe AWS Certified AI Practitioner exam covers a wide range of topics related to AI, ML, and generative AI technologies, including:\n\n- General AI/ML concepts\n- Generative AI techniques and applications\n- AWS services and tools for AI and ML\n\nOverall, the AWS Certified AI Practitioner certification is designed to validate an individual's ability to apply AI/ML and generative AI technologies in real-world scenarios, independent of a specific job role.",
    },
]

filename = "cached_contents.csv"

with open(filename, "w", newline="") as csvfile:
    fieldnames = ["question", "answer"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
    print(f"Cached contents written to: {filename}!")
