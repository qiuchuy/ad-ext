from fastapi import FastAPI, HTTPException
import ailang # Assuming ailang is a library you are using
import uvicorn
import io
import contextlib
import traceback
import signal
from pydantic import BaseModel
import base64 # Ensure base64 is imported

# Import pygraphviz
import pygraphviz as pgv

import ailang as al
import torch # Assuming torch is a library you are using
# import ailang as al # You already imported ailang, this might be redundant unless it's a submodule

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
        # Note: signal.SIGALRM is not available on Windows.
        # Consider alternative timeout mechanisms if cross-platform compatibility is critical.
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(request.timeout)
        except AttributeError:
            print("Warning: signal.SIGALRM is not available on this platform (e.g., Windows). Timeout will not be enforced by signal.")


        try:
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                # Create a local environment but allow global imports
                local_vars = {}
                # Use globals() to allow importing external libraries
                try:
                    code_to_execute = request.code
                    exec(code_to_execute, globals(), local_vars)
                except Exception as e:
                    print(f"An error occurred during exec: {e}")
                result = local_vars.get('result', None)

            # Disable the alarm
            try:
                signal.alarm(0)
            except AttributeError:
                pass # SIGALRM not available


            # Return the captured output and result
            returned_payload = {
                "success": True,
                "output": output_buffer.getvalue(),
                "result": result
            }
            if request.task == "autodiff":
                import subprocess # Keep for postprocessing.py if it doesn't generate dot files directly

                fwd_dot_file_name = "forward" # Just the name, no extension initially
                bwd_dot_file_name = "backward"

                # Assuming postprocessing.py generates/modifies .dot files
                # If postprocessing.py directly outputs DOT content, you might adjust this.
                try:
                    subprocess.check_call(["python", "postprocessing.py", "--input", fwd_dot_file_name])
                    subprocess.check_call(["python", "postprocessing.py", "--input", bwd_dot_file_name])
                except subprocess.CalledProcessError as e:
                    error_msg = f"Error during postprocessing.py execution: {e}\nOutput: {e.output}\nStderr: {e.stderr}"
                    print(error_msg)
                    returned_payload.update({
                        "success": False,
                        "error": error_msg,
                        "fwd_graph": None,
                        "bwd_graph": None
                    })
                    return returned_payload
                except FileNotFoundError:
                    error_msg = "Error: postprocessing.py not found. Ensure it is in the correct path."
                    print(error_msg)
                    returned_payload.update({
                        "success": False,
                        "error": error_msg,
                        "fwd_graph": None,
                        "bwd_graph": None
                    })
                    return returned_payload


                fwd_graph_svg_str = None
                bwd_graph_svg_str = None

                # --- PyGraphviz rendering ---
                for i, dot_file_basename in enumerate([fwd_dot_file_name, bwd_dot_file_name]):
                    dot_file_path = f"{dot_file_basename}" # Assuming postprocessing.py adds .dot
                    try:
                        # Create a graph from the .dot file
                        graph = pgv.AGraph(dot_file_path, strict=False, directed=True)
                        graph.graph_attr['ranksep'] = '0' # Set ranksep to 1.0 inches
                        graph.graph_attr['newrank'] = True # Set ranksep to 1.0 inches
                        layout_engine = 'dot'
                        graph.layout(prog=layout_engine) # Layout the graph using 'dot'
                        
                        # Render to SVG string in memory
                        svg_buffer = io.BytesIO()
                        graph.draw(svg_buffer, format='svg', prog=layout_engine)
                        svg_buffer.seek(0)
                        svg_content = svg_buffer.read()
                        
                        if i == 0:
                            fwd_graph_svg_str = base64.b64encode(svg_content).decode('utf-8')
                        else:
                            bwd_graph_svg_str = base64.b64encode(svg_content).decode('utf-8')

                    except FileNotFoundError:
                        err_msg = f"Error: DOT file '{dot_file_path}' not found after postprocessing."
                        print(err_msg)
                        if i == 0: fwd_graph_svg_str = f"Error: {err_msg}"
                        else: bwd_graph_svg_str = f"Error: {err_msg}"
                    except Exception as e:
                        err_msg = f"Error rendering {dot_file_path} with pygraphviz: {str(e)}"
                        print(err_msg)
                        # You might want to return an error or a placeholder image/message
                        if i == 0: fwd_graph_svg_str = f"Error: {err_msg}"
                        else: bwd_graph_svg_str = f"Error: {err_msg}"
                # --- End PyGraphviz rendering ---

                # Update returned_payload after encoding both graphs
                returned_payload.update({
                    "fwd_graph": fwd_graph_svg_str,
                    "bwd_graph": bwd_graph_svg_str
                })
            return returned_payload
        except TimeoutException:
            return {
                "success": False,
                "output": output_buffer.getvalue(), # Output might still be valuable
                "error": "Execution timed out"
            }
        except Exception as e:
            error_msg = traceback.format_exc()
            return {
                "success": False,
                "output": output_buffer.getvalue(), # Output might still be valuable
                "error": error_msg
            }
        finally:
            # Ensure alarm is disabled
            try:
                signal.alarm(0)
            except AttributeError:
                pass # SIGALRM not available

    except Exception as e:
        # This is a catch-all for errors outside the main try-finally (e.g., issues with FastAPI itself)
        # or if output_buffer was not initialized, etc.
        raise HTTPException(status_code=500, detail=f"Outer exception: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    # Run the server on 0.0.0.0 to make it accessible outside the container
    # Use port 3389 which is mapped to host's 8888
    uvicorn.run(app, host="0.0.0.0", port=3389)
