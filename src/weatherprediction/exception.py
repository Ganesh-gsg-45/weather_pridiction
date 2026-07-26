import sys


def error_message_detail(error, error_detail: sys) -> str:
    """
    Build a detailed error message with file name and line number
    where the exception was raised.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    message = (
        f"Error occurred in Python script: [{file_name}] "
        f"at line [{line_number}] — {str(error)}"
    )
    return message


class WeatherException(Exception):
    """Custom exception for the Weather Prediction pipeline."""

    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self) -> str:
        return self.error_message
