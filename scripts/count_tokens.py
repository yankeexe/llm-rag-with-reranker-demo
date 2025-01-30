"""
Helper module to count the number of tokens before passing it to the embedding model.

Attributes:
    content (str): Sample text content about AWS AI/ML candidate requirements
    tokenizer: Pre-trained tokenizer model from Nomic AI
    tokens (list): Tokenized text content
    token_count (int): Number of tokens in the content
"""

from transformers import AutoTokenizer

content = """
The target candidate is expected to have a specific set of skills and experience in the field of Artificial Intelligence (AI) and Machine Learning (ML), with a focus on their use of AI/ML technologies within Amazon Web Services (AWS). The key characteristics of the target candidate include:\n\n- **Exposure to AI/ML technologies**: The candidate has up to 6 months of exposure to AI/ML technologies on AWS, but may not necessarily be building AI/ML solutions from scratch.\n- **Responsibility vs. Building**: The candidate uses AI/ML technologies in their work, but may not always be responsible for building or implementing these solutions.\n- **AWS Familiarity**: The candidate has a good understanding of the core AWS services (e.g., Amazon EC2, Amazon S3, AWS Lambda, and Amazon SageMaker) and their respective use cases.\n\nOverall, the target candidate is expected to demonstrate a solid foundation in AI/ML concepts and their application within AWS, while also being able to apply this knowledge to real-world scenarios.
"""
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
tokens = tokenizer.tokenize(content)
token_count = len(tokens)
print(token_count)
