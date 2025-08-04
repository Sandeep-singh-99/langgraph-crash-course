from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writting execllent twitter posts."
            "Generate the best twitter post possible based for the user's request."
            "If the user provides critique, respond with a revised version of you previous attempt."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a virtual influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, variety, style, etc."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
