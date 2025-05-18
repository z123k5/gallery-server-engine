from enum import Enum
import threading
globalVarUseLock = threading.Lock()
dbLock = threading.Lock()
vlmModelLock = threading.Lock()

# Server log id in db
class LOGI(Enum):
    EXCEPTION = 1   # When exception occurs
    INFO = 2        # When some tips occurs
    WARNING = 3     # When some warning occurs

    USER_LOGIN = 4   # When user login
    USER_LOGOUT = 5  # When user logout
    USER_REGISTER = 6  # When user register
    USER_DELETE = 7  # When user delete
    USER_RESOLVE = 8  # When user resolve one media
    USER_QUERY = 9  # When user query some media


    ADMIN_SUPERVISE_USER_REGISTER = 12  # When admin supervise user register
    ADMIN_CLEAR_USER_PASSWORD = 13  # When admin clear user password
    ADMIN_DELETE_USER = 14  # When admin delete user
    ADMIN_LIMIT_USER_FREQUENCY = 15  # When admin limit user frequency
    ADMIN_MODIFY_USER_CLOUD = 16  # When admin modify user cloud
    ADMIN_CHANGE_MODEL = 17  # When admin change model
    ADMIN_MODIFY_USER_MEDIA = 18  # When admin modify user media
    ADMIN_LIMIT_USER_MEDIA_COUNTS = 19  # When admin limit user media counts





