from typing import Any, Coroutine, List, Literal, Optional, Union, overload

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import ThoughtStep
from approaches.chatapproach import ChatApproach
from approaches.promptmanager import PromptManager
from core.authentication import AuthenticationHelper
import pyodbc

class ChatReadSQLDataApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
        sql_server_db_connection_string: str,
        prompt_manager: PromptManager
    ):
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model, default_to_minimum=self.ALLOW_NON_GPT_MODELS)
        self.prompt_manager = prompt_manager
        self.query_rewrite_tools = self.prompt_manager.load_tools("chat_query_rewrite_tools.json")
        self.answer_prompt = self.prompt_manager.load_prompt("chat_answer_question.prompty")
        self.sql_generation_prompt = self.prompt_manager.load_prompt("chat_generate_sql.prompty")
        self.sql_server_db_connection_string = sql_server_db_connection_string

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[False],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, ChatCompletion]]: ...


    def get_generated_sql(self, chat_completion: ChatCompletion, original_user_query: str) -> str:
        """
        Extracts the generated SQL query from the chat completion response.
        Assumes that the model outputs a structured SQL query in its response.

        Args:
            chat_completion (ChatCompletion): The OpenAI response containing the SQL query.
            original_user_query (str): The original user question.

        Returns:
            str: The generated SQL query.
        """
        # Extract the first choice from the completion response
        message_content = chat_completion.choices[0].message.content.strip()

        # Optional: Ensure the model-generated response is valid SQL
        if not message_content.lower().startswith("select") and "from" not in message_content.lower():
            raise ValueError(f"Generated response is not a valid SQL query: {message_content}")

        return message_content

    async def execute_sql(self, query: str):
        # Define your connection parameters
 
        try:
               # Establish a connection
            conn = pyodbc.connect(self.sql_server_db_connection_string)

            # Create a cursor object
            cursor = conn.cursor()
            cursor.execute(query)

            columns = [desc[0] for desc in cursor.description]  # Get column names

            # Fetch results
            rows = cursor.fetchall()
           # result = "\n".join([str(row) for row in rows])  # Convert rows to a formatted string
            # Format each row as a string: "(column1:value1, column2:value2, ...)"
            formatted_rows = [f"({', '.join(f'{col}:{val}' for col, val in zip(columns, row))})" for row in rows]

            # Join all formatted rows into a single string
            result = ", ".join(formatted_rows)
            if cursor is not None:
                cursor.close()

            if conn is not None:
                conn.close()

        except pyodbc.Error as e:
            result = f"Error executing SQL query"
        return result


    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[True],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]]]: ...

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]]]:
        seed = overrides.get("seed", None)

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")

        rendered_query_prompt = self.prompt_manager.render_prompt(
            self.sql_generation_prompt, {"user_query": original_user_query, "past_messages": messages[:-1]}
        )
        #tools: List[ChatCompletionToolParam] = self.query_rewrite_tools

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        query_response_token_limit = 1000
        query_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=rendered_query_prompt.system_content,
            few_shots=rendered_query_prompt.few_shot_messages,
            past_messages=rendered_query_prompt.past_messages,
            new_user_content=rendered_query_prompt.new_user_content,
           # tools=tools,
            max_tokens=self.chatgpt_token_limit - query_response_token_limit,
            fallback_to_default=self.ALLOW_NON_GPT_MODELS,
        )

        chat_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            messages=query_messages,  # type: ignore
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            temperature=0.0,  # Minimize creativity for search query generation
            max_tokens=query_response_token_limit,  # Setting too low risks malformed JSON, setting too high may affect performance
            n=1,
          #  tools=tools,
            seed=seed,
        )
        
        # STEP 2: Generate SQL and execute it to retrieve search results
        generated_sql = self.get_generated_sql(chat_completion, original_user_query)
        results = await self.execute_sql(generated_sql)

        response_token_limit = 1024

        if(results == "Error executing SQL query" or results == ""):
            messages = build_messages(
                model=self.chatgpt_model,
                system_prompt='You are a helpful AI assistant, but you couldn''t find the answer in the database, say that you do not know the answer.',
                past_messages=rendered_query_prompt.past_messages,
                new_user_content=original_user_query,
                max_tokens=self.chatgpt_token_limit - response_token_limit,
                fallback_to_default=self.ALLOW_NON_GPT_MODELS,
            )
        else:
            messages = build_messages(
                model=self.chatgpt_model,
                system_prompt='You are a helpful assistant. When you receive the User Question and Answer, rephrase the Answer to make it clear to the user, don''t hallucinate and add random data to the results, your results should be restricted by the answer you received.',
                past_messages=rendered_query_prompt.past_messages,
                new_user_content='User Question: ' + original_user_query + ', Answer: ' + results,
                max_tokens=self.chatgpt_token_limit - response_token_limit,
                fallback_to_default=self.ALLOW_NON_GPT_MODELS,
            )

        thought_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt='Generate SQL query for the User Question',
            few_shots=rendered_query_prompt.few_shot_messages,
            past_messages=rendered_query_prompt.past_messages,
            new_user_content=rendered_query_prompt.new_user_content,
            max_tokens=self.chatgpt_token_limit - query_response_token_limit,
            fallback_to_default=self.ALLOW_NON_GPT_MODELS,
        )

        extra_info = {
           "data_points": {"text": ["source1", "source2","source3"]},
            "thoughts": [
                ThoughtStep(
                    "Prompt to generate SQL query",
                    thought_messages,
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    messages,
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        chat_coroutine = self.openai_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            stream=should_stream,
            seed=seed,
        )
        return (extra_info, chat_coroutine)
