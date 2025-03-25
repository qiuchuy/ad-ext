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
import ailang as al

app = FastAPI()

class ScriptRequest(BaseModel):
    code: str
    timeout: int = 30  # Default timeout in seconds
    task: str

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
                try:
                    code_to_execute = request.code
                    exec(code_to_execute, globals(), local_vars)
                except Exception as e:
                    print(f"An error occurred: {e}")
                result = local_vars.get('result', None)
                
            # Disable the alarm
            signal.alarm(0)
            
            # Return the captured output and result
            returned_payload = {
                "success": True,
                "output": output_buffer.getvalue(),
                "result": result
            }
            if request.task == "autodiff":
                import subprocess
                import base64
                fwd_dot_file = "forward"
                bwd_dot_file = "backward"
                subprocess.check_call(["python", "postprocessing.py", "--input", fwd_dot_file])
                subprocess.check_call(["python", "postprocessing.py", "--input", bwd_dot_file])
                fwd_png_graph = subprocess.check_output(["dot", "-Tsvg", fwd_dot_file])
                bwd_png_graph = subprocess.check_output(["dot", "-Tsvg", bwd_dot_file])
                fwd_graph = base64.b64encode(fwd_png_graph).decode('utf-8')
                bwd_graph = base64.b64encode(bwd_png_graph).decode('utf-8')
                # Update returned_payload after encoding both graphs
                returned_payload.update({
                    "fwd_graph": fwd_graph,
                    "bwd_graph": bwd_graph
                })
            return returned_payload
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
