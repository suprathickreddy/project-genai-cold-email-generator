import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
        You are Srinathareddy Suprathickreddy, currently pursuing a Master’s in Data Analytics at Clark University. With a strong foundation in data science, machine learning, and AI, along with hands-on project experience,
        excited about the opportunity to contribute to your team at [Company Name] 
        At Clark University, I have deepened my knowledge of data analytics and honed my skills in Python, machine learning, SQL, and deep learning techniques, which align well with the demands of modern AI-driven enterprises.
        I am confident in my ability to help your team leverage data and AI technologies to foster scalability, optimize processes, reduce costs, and enhance overall efficiency. 
        Your job is to write a cold email to the hiring manager regarding the job mentioned above describing the capability of me 
        in fulfilling their needs.
        also add my linkedin [https://www.linkedin.com/in/suprathickreddy/] and github links [https://github.com/suprathickreddy] to show my profile details
        Remember you are suprathickreddy, Masters in Data Analytics at Clark University. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))