from typing import Any, Optional

class AI_APICallError(Exception):
    """
    Custom error class for API request failures.
    
    Attributes:
        url (str): The URL of the API request that failed
        request_body_values (dict): The request body values sent to the API
        status_code (int): The HTTP status code returned by the API
        response_headers (dict): The response headers returned by the API
        response_body (Any): The response body returned by the API
        is_retryable (bool): Whether the request can be retried based on the status code
        data (dict): Any additional data associated with the error
    """
    
    def __init__(
        self,
        url: str,
        request_body_values: dict,
        status_code: int,
        response_headers: dict,
        response_body: any,
        is_retryable: bool,
        data: dict = None
    ):
        self.url = url
        self.request_body_values = request_body_values
        self.status_code = status_code
        self.response_headers = response_headers
        self.response_body = response_body
        self.is_retryable = is_retryable
        self.data = data or {}
        
        # Create a descriptive error message
        message = f"API request to {url} failed with status code {status_code}"
        super().__init__(message)
    
    def __str__(self) -> str:
        """Return a string representation of the error."""
        return (
            f"APIError: {self.args[0]}\n"
            f"URL: {self.url}\n"
            f"Status Code: {self.status_code}\n"
            f"Is Retryable: {self.is_retryable}\n"
            f"Response Body: {self.response_body}"
        )

class AI_UnsupportedFunctionalityError(Exception):
    """
    Custom error class for unsupported functionality.
    
    Attributes:
        functionality (str): The name of the unsupported functionality
        message (str): Custom error message (optional)
    """
    
    def __init__(
        self,
        functionality: str,
        message: str = None
    ):
        self.functionality = functionality
        self.message = message or f"'{functionality}' functionality not supported."
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return a string representation of the error."""
        return f"UnsupportedFunctionalityError: {self.message}"

class AI_ToolExecutionError(Exception):
    """
    Custom error class for tool execution failures.
    
    Attributes:
        tool_name (str): The name of the tool that failed
        tool_args (Any): The arguments passed to the tool
        tool_call_id (str): The unique identifier for this tool call
        message (str): Error message
        cause (Exception, optional): The underlying cause of the error
    """
    def __init__(
        self,
        tool_name: str,
        tool_args: Any,
        tool_call_id: str,
        message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_call_id = tool_call_id
        self.cause = cause
        
        if message is None:
            message = f"Error executing tool {tool_name}: {str(cause) if cause else 'unknown error'}"
        
        super().__init__(message)
    
    def __str__(self) -> str:
        details = [
            f"Tool Name: {self.tool_name}",
            f"Tool Call ID: {self.tool_call_id}",
            f"Tool Args: {self.tool_args}",
        ]
        if self.cause:
            details.append(f"Cause: {str(self.cause)}")
            
        return f"{super().__str__()}\n" + "\n".join(details)
