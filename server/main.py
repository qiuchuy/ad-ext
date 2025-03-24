from fastapi import FastAPI, HTTPException
import ailang
import uvicorn
import io
import contextlib
import traceback
import signal
from pydantic import BaseModel

import torch
import ailang

app = FastAPI()

class ScriptRequest(BaseModel):
    code: str
    timeout: int = 30  # Default timeout in seconds

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/execute")
async def execute_code(request: ScriptRequest):
    try:
        # Capture stdout and stderr
        output_buffer = io.StringIO()
        
        # Setup timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(request.timeout)
        
        try:
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                # Create a local environment but allow global imports
                local_vars = {}
                # Use globals() to allow importing external libraries
                exec(request.code, globals(), local_vars)
                result = local_vars.get('result', None)
                
            # Disable the alarm
            signal.alarm(0)
            
            # Return the captured output and result
            return {
                "success": True,
                "output": output_buffer.getvalue(),
                "result": result
            }
        except TimeoutException:
            return {
                "success": False,
                "output": output_buffer.getvalue(),
                "error": "Execution timed out"
            }
        except Exception as e:
            error_msg = traceback.format_exc()
            return {
                "success": False,
                "output": output_buffer.getvalue(),
                "error": error_msg
            }
        finally:
            # Ensure alarm is disabled
            signal.alarm(0)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server on 0.0.0.0 to make it accessible outside the container
    # Use port 3389 which is mapped to host's 8888
    uvicorn.run(app, host="0.0.0.0", port=3389)
