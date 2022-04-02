
# coding: utf-8

# In[11]:

from datetime import datetime


# In[9]:

class ErrorLogger():
    def __init__():
        pass
    
    def log_error(message="", filepath="", write_mode="a"):
        message_formatted = str(datetime.now()) + " : " + str(message) + "\n\n"
        
        with open(filepath, write_mode) as my_logger_file:
            my_logger_file.write(message_formatted)
