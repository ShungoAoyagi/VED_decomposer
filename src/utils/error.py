from typing import Optional, List, Dict, Any, Union
import logging
import warnings
from enum import Enum, auto

# setting logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# define error level
class ErrorLevel(Enum):
    CRITICAL = auto()  # throw an exception as an anomalous error
    ERROR = auto()     # record as an error even though it is normal
    WARNING = auto()   # record as a warning

class ErrorCode(Enum):
    NOT_FOUND = "Not found error"
    VALIDATION = "Validation error"
    INVALID_INPUT = "Invalid input error"

# base custom error class
class CustomError(Exception):
    """Base custom error class"""
    
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.message: str = message
        self.error_code: str = error_code
        self.details: Dict[str, Any] = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Return error information in dictionary format"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }

# anomalous error (exceptional error)
class CriticalError(CustomError):
    """Critical error that the system cannot continue"""
    pass

# domain-specific error example
class ValidationError(CustomError):
    """Error when the input value fails to validate"""
    pass

class DatabaseError(CustomError):
    """Error when the database operation fails"""
    pass

class ErrorHandler:
    """Handler that manages errors and warnings"""
    
    def __init__(self) -> None:
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
    
    def handle(self, 
               message: str, 
               error_code: ErrorCode, 
               level: ErrorLevel, 
               details: Optional[Dict[str, Any]] = None) -> Union[None, CustomError]:
        """
        Process errors based on the error level
        
        Args:
            message: error message
            error_code: error code
            level: error level
            details: additional details
            
        Returns:
            Returns a CustomError object if the level is CRITICAL, otherwise None
        """
        error_info = {
            "message": message,
            "error_code": error_code,
            "details": details or {}
        }
        
        if level == ErrorLevel.CRITICAL:
            # anomalous: select the appropriate error type and create an exception
            if error_code == ErrorCode.VALIDATION:
                return ValidationError(message, error_code, details)
            elif error_code == ErrorCode.DATABASE:
                return DatabaseError(message, error_code, details)
            else:
                return CriticalError(message, error_code, details)
        
        elif level == ErrorLevel.ERROR:
            # normal: record as an error
            self.errors.append(error_info)
            logger.error(f"[{error_code}] {message}")
        
        elif level == ErrorLevel.WARNING:
            # normal: record as a warning
            self.warnings.append(error_info)
            logger.warning(f"[{error_code}] {message}")
            warnings.warn(f"[{error_code}] {message}")
        
        return None
    
    def has_errors(self) -> bool:
        """Check if errors are recorded"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if warnings are recorded"""
        return len(self.warnings) > 0
    
    def get_all_issues(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all errors and warnings"""
        return {
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def clear(self) -> None:
        """Clear all errors and warnings"""
        self.errors = []
        self.warnings = []


# 使用例
def process_data(data: Dict[str, Any], error_handler: ErrorHandler) -> Dict[str, Any]:
    """
    データ処理関数の例
    
    Args:
        data: 処理するデータ
        error_handler: エラーハンドラー
        
    Returns:
        処理結果
    """
    result = {"success": True, "processed_data": None}
    
    # 入力検証の例
    if "name" not in data:
        # 重大なエラー（例外を発生させる）
        error = error_handler.handle(
            "名前フィールドは必須です", 
            "VAL001", 
            ErrorLevel.CRITICAL
        )
        if error:
            raise error
    
    if "age" not in data:
        # エラーとして記録するが処理は続行
        error_handler.handle(
            "年齢フィールドがありません", 
            "VAL002", 
            ErrorLevel.ERROR,
            {"available_fields": list(data.keys())}
        )
        result["success"] = False
    
    if "email" in data and not data["email"].endswith((".com", ".jp")):
        # 警告として記録
        error_handler.handle(
            "メールアドレスの形式が一般的ではありません",
            "VAL003",
            ErrorLevel.WARNING,
            {"provided_email": data.get("email")}
        )
    
    # 正常に処理できた場合
    if result["success"]:
        result["processed_data"] = {
            "name": data.get("name", ""),
            "age": data.get("age", 0),
            "email": data.get("email", "")
        }
    
    return result
