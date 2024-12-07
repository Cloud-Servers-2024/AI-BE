from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="School work copilot")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "code": 200}


class agent_output(BaseModel):
    response_to_student: str = Field(description="The message you are giving to the student")


school_work_copilot_agent = ChatPromptTemplate.from_messages(
            [
                (
                    "You are a school work copilot that specializes in helping students complete their assignments"
                    "You will receive a message from a student asking for help with an assignment or questions about the assignment"
                    "Your job is to help the student complete the assignment or answer the questions about the assignment"
                    "They may also ask for context on the topic of the assignment and you should provide that as well"
                    "You will be given an assignment title/description as additional context"
                ),
                (
                    "user",
                    "<User Message>"
                    "{user_message}"
                    "<User Message>"
                    "<Assignment Title/Description>"
                    "{assignment_title_description}"
                    "<Assignment Title/Description>"
                ),
            ]
)


class SchoolWorkCopilotAgent():
    def __init__(self):
        #TODO: DELETE THIS KEY AFTER DONE WITH DEMO
        self.open_ai_key = "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"
        self.llm_model = "gpt-4o"

    def generate_response_to_student(self, user_message, assignment_title_description):
        llm = ChatOpenAI(api_key=str(self.open_ai_key), model=str(self.llm_model))
        chain = school_work_copilot_agent | llm.with_structured_output(schema=agent_output)
        response_to_student = chain.invoke({
            "user_message": user_message,
            "assignment_title_description": assignment_title_description
        })
        return response_to_student


class API_schema(BaseModel):
    user_message: str = Field(description="The message from the student asking for help with an assignment or questions about the assignment")
    assignment_title_description: str = Field(description="The assignment title/description as additional context")


def initialize_agent():
    try:
        agent = SchoolWorkCopilotAgent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        raise e
    return agent


@app.post("/generate_AI_response")
async def generate_AI_response(message: API_schema):
    try:
        agent = initialize_agent()
        response_to_student = agent.generate_response_to_student(message.user_message, message.assignment_title_description)
        return {"response_to_student": response_to_student}
    except Exception as e:
        return {"error": f"Error generating AI response: {e}"}


#to run the server use python main.py, be sure to activate the venv first
#TODO, dockerize the app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

