import argparse
import asyncio
import json
import os
import pathlib

import aiohttp
import gradio as gr
import requests
import asyncio
import aiohttp
import re


host_url = "127.0.0.1"

models = []
draft_models = []
loras = []
templates = []
overrides = []

model_load_task = None
model_load_state = False
download_task = None

parser = argparse.ArgumentParser(description="TabbyAPI Gradio Loader")
parser.add_argument(
    "-p",
    "--port",
    type=int,
    default=7860,
    help="Specify port to host the WebUI on (default 7860)",
)
parser.add_argument(
    "-l", "--listen", action="store_true", help="Share WebUI link via LAN"
)
parser.add_argument(
    "-n",
    "--noauth",
    action="store_true",
    help="Specify TabbyAPI endpoint that has no authorization",
)
parser.add_argument(
    "-s",
    "--share",
    action="store_true",
    help="Share WebUI link remotely via Gradio's built in tunnel",
)
parser.add_argument(
    "-a",
    "--autolaunch",
    action="store_true",
    help="Launch browser after starting WebUI",
)
parser.add_argument(
    "-e",
    "--endpoint_url",
    type=str,
    default="http://localhost:5000",
    help="TabbyAPI endpoint URL (default http://localhost:5000)",
)
parser.add_argument(
    "-k",
    "--admin_key",
    type=str,
    default=None,
    help="TabbyAPI admin key, connect automatically on launch",
)
args = parser.parse_args()
if args.listen:
    host_url = "0.0.0.0"

if "TABBY_ADMIN_KEY" in os.environ and not args.admin_key:
    args.admin_key = os.environ["TABBY_ADMIN_KEY"]
conn_url = None
args.admin_key = args.admin_key


def read_preset(name):
    if not name:
        raise gr.Error("Please select a preset to load.")
    path = pathlib.Path(f"./presets/{name}.json").resolve()
    with open(path, "r") as openfile:
        data = json.load(openfile)
    gr.Info(f"Preset {name} loaded.")
    return (
        gr.Dropdown(value=data.get("name")),
        gr.Number(value=data.get("max_seq_len")),
        gr.Number(value=data.get("override_base_seq_len")),
        gr.Number(value=data.get("cache_size")),
        gr.Checkbox(value=data.get("gpu_split_auto")),
        gr.Textbox(value=data.get("gpu_split")),
        gr.Number(value=data.get("rope_scale")),
        gr.Number(value=data.get("rope_alpha")),
        gr.Radio(value=data.get("cache_mode")),
        gr.Dropdown(value=data.get("prompt_template")),
        gr.Number(value=data.get("num_experts_per_token")),
        gr.Dropdown(value=data.get("draft_model_name")),
        gr.Number(value=data.get("draft_rope_scale")),
        gr.Number(value=data.get("draft_rope_alpha")),
        gr.Radio(value=data.get("draft_cache_mode")),
        gr.Checkbox(value=data.get("fasttensors")),
        gr.Textbox(value=data.get("autosplit_reserve")),
        gr.Number(value=data.get("chunk_size")),
    )


def del_preset(name):
    if not name:
        raise gr.Error("Please select a preset to delete.")
    path = pathlib.Path(f"./presets/{name}.json").resolve()
    path.unlink()
    gr.Info(f"Preset {name} deleted.")
    return get_preset_list()


def write_preset(
    name,
    model_name,
    max_seq_len,
    override_base_seq_len,
    cache_size,
    gpu_split_auto,
    gpu_split,
    model_rope_scale,
    model_rope_alpha,
    cache_mode,
    prompt_template,
    num_experts_per_token,
    draft_model_name,
    draft_rope_scale,
    draft_rope_alpha,
    draft_cache_mode,
    fasttensors,
    autosplit_reserve,
    chunk_size,
):
    if not name:
        raise gr.Error("Please enter a name for your new preset.")
    path = pathlib.Path(f"./presets/{name}.json").resolve()
    data = {
        "name": model_name,
        "max_seq_len": max_seq_len,
        "override_base_seq_len": override_base_seq_len,
        "cache_size": cache_size,
        "gpu_split_auto": gpu_split_auto,
        "gpu_split": gpu_split,
        "rope_scale": model_rope_scale,
        "rope_alpha": model_rope_alpha,
        "cache_mode": cache_mode,
        "prompt_template": prompt_template,
        "num_experts_per_token": num_experts_per_token,
        "draft_model_name": draft_model_name,
        "draft_rope_scale": draft_rope_scale,
        "draft_rope_alpha": draft_rope_alpha,
        "draft_cache_mode": draft_cache_mode,
        "fasttensors": fasttensors,
        "autosplit_reserve": autosplit_reserve,
        "chunk_size": chunk_size,
    }
    with open(path, "w") as outfile:
        json.dump(data, outfile, indent=4)
    gr.Info(f"Preset {name} saved.")
    return gr.Textbox(value=None), get_preset_list()


def get_preset_list(raw=False):
    preset_path = pathlib.Path("./presets").resolve()
    preset_list = []
    for path in preset_path.iterdir():
        if path.is_file() and path.name.endswith(".json"):
            preset_list.append(path.stem)
    preset_list.sort(key=str.lower)
    if raw:
        return preset_list
    return gr.Dropdown(choices=[""] + preset_list, value=None)


def connect(api_url, admin_key, silent=False):
    global conn_url
    global models
    global draft_models
    global loras
    global templates
    global overrides

    if not args.noauth:
        try:
            a = requests.get(
                url=api_url + "/v1/auth/permission", headers={"X-api-key": admin_key}
            )
            a.raise_for_status()
            if a.json().get("permission") != "admin":
                raise ValueError(
                    "The provided authentication key must be an admin key to access the loader's functions."
                )
        except Exception as e:
            raise gr.Error(e)

    try:
        m = requests.get(
            url=api_url + "/v1/model/list", headers={"X-api-key": admin_key}
        )
        m.raise_for_status()
        d = requests.get(
            url=api_url + "/v1/model/draft/list", headers={"X-api-key": admin_key}
        )
        d.raise_for_status()
        lo = requests.get(
            url=api_url + "/v1/lora/list", headers={"X-api-key": admin_key}
        )
        lo.raise_for_status()
        t = requests.get(
            url=api_url + "/v1/template/list", headers={"X-api-key": admin_key}
        )
        t.raise_for_status()
        so = requests.get(
            url=api_url + "/v1/sampling/override/list", headers={"X-api-key": admin_key}
        )
        so.raise_for_status()
    except Exception as e:
        raise gr.Error(e)

    conn_url = api_url
    args.admin_key = admin_key

    models = []
    for model in m.json().get("data"):
        models.append(model.get("id"))
    models.sort(key=str.lower)

    draft_models = []
    for draft_model in d.json().get("data"):
        draft_models.append(draft_model.get("id"))
    draft_models.sort(key=str.lower)

    loras = []
    for lora in lo.json().get("data"):
        loras.append(lora.get("id"))
    loras.sort(key=str.lower)

    templates = []
    for template in t.json().get("data"):
        templates.append(template)
    templates.sort(key=str.lower)

    overrides = []
    for override in so.json().get("presets"):
        overrides.append(override)
    overrides.sort(key=str.lower)

    if not silent:
        print("TabbyAPI connected.")

    return models, draft_models, loras, templates, overrides


def disconnect():
    global conn_url
    global models
    global draft_models
    global loras
    global templates
    global overrides
    conn_url = None
    args.admin_key = None
    models = []
    draft_models = []
    loras = []
    templates = []
    overrides = []
    return


def get_model_list():
    return gr.Dropdown(choices=[""] + models, value=None)


def get_draft_model_list():
    return gr.Dropdown(choices=[""] + draft_models, value=None)


def get_lora_list():
    return gr.Dropdown(choices=loras, value=[])


def get_template_list():
    return gr.Dropdown(choices=[""] + templates, value=None)


def get_override_list():
    return gr.Dropdown(choices=[""] + overrides, value=None)


def get_current_model():
    model_card = requests.get(
        url=conn_url + "/v1/model", headers={"X-api-key": args.admin_key}
    ).json()
    if not model_card.get("id"):
        return gr.Textbox(value=None)
    params = model_card.get("parameters")
    draft_model_card = params.get("draft")
    model = f'{model_card.get("id")} (context: {params.get("max_seq_len")}, cache size: {params.get("cache_size")}, rope scale: {params.get("rope_scale")}, rope alpha: {params.get("rope_alpha")})'

    if draft_model_card:
        draft_params = draft_model_card.get("parameters")
        model += f' | {draft_model_card.get("id")} (rope scale: {draft_params.get("rope_scale")}, rope alpha: {draft_params.get("rope_alpha")})'
    return gr.Textbox(value=model)


def get_current_loras():
    lo = requests.get(
        url=conn_url + "/v1/lora", headers={"X-api-key": args.admin_key}
    ).json()
    if not lo.get("data"):
        return gr.Textbox(value=None)
    lora_list = lo.get("data")
    loras = []
    for lora in lora_list:
        loras.append(f'{lora.get("id")} (scaling: {lora.get("scaling")})')
    return gr.Textbox(value=", ".join(loras))


def update_loras_table(loras):
    array = []
    for lora in loras:
        array.append(1.0)
    if array:
        return gr.List(
            value=[array],
            col_count=(len(array), "fixed"),
            row_count=(1, "fixed"),
            headers=loras,
            visible=True,
        )
    else:
        return gr.List(value=None, visible=False)


def format_list_param(param):
    if isinstance(param, str):
        return [float(x.strip()) for x in param.split(",") if x.strip()]
    elif isinstance(param, (int, float)):
        return [float(param)]
    elif isinstance(param, list):
        return [float(x) for x in param]
    return None


print(f"Admin Key from Args: {args.admin_key}")
if args.noauth:
    print("Using no-auth mode.")
else:
    print(f"Conn Key: {args.admin_key}")


async def test_connection():
    async with aiohttp.ClientSession() as session:
        response = await session.get(
            url=conn_url + "/v1/model", headers={"X-api-key": args.admin_key}
        )
        print(f"Test connection status: {response.status}")
        print(f"Test connection body: {await response.text()}")


async def load_model(
    model_name,
    max_seq_len,
    override_base_seq_len,
    cache_size,
    gpu_split_auto,
    gpu_split,
    model_rope_scale,
    model_rope_alpha,
    cache_mode,
    prompt_template,
    num_experts_per_token,
    draft_model_name,
    draft_rope_scale,
    draft_rope_alpha,
    draft_cache_mode,
    fasttensors,
    autosplit_reserve,
    chunk_size,
):
    global model_load_task
    global model_load_state
    global conn_url

    await test_connection()  # TODO: removeme when debugging done

    model_load_state = True
    if not model_name:
        raise gr.Error("Specify a model to load!")

    gr.Info(f"Using connection URL: {conn_url}")
    gr.Info(f"API Key (first 2 chars): {args.admin_key[:2]}...")

    async def attempt_load(session, seq_len, c_size, attempt_number):
        nonlocal max_seq_len, cache_size

        formatted_gpu_split = (
            format_list_param(gpu_split) if not gpu_split_auto else None
        )
        formatted_autosplit_reserve = format_list_param(autosplit_reserve)

        request = {
            "name": model_name,
            "max_seq_len": seq_len,
            "override_base_seq_len": override_base_seq_len,
            "cache_size": c_size,
            "gpu_split_auto": gpu_split_auto,
            "gpu_split": formatted_gpu_split,
            "rope_scale": model_rope_scale,
            "rope_alpha": model_rope_alpha,
            "cache_mode": cache_mode,
            "prompt_template": prompt_template,
            "num_experts_per_token": num_experts_per_token,
            "fasttensors": fasttensors,
            "autosplit_reserve": formatted_autosplit_reserve,
            "chunk_size": chunk_size,
            "draft": (
                {
                    "draft_model_name": draft_model_name,
                    "draft_rope_scale": draft_rope_scale,
                    "draft_rope_alpha": draft_rope_alpha,
                    "draft_cache_mode": draft_cache_mode,
                }
                if draft_model_name
                else None
            ),
        }
        request = {k: v for k, v in request.items() if v is not None}

        gr.Info(
            f"Attempt {attempt_number}: Sending request: {json.dumps(request, indent=2)}"
        )

        try:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
            async with session.post(
                url=conn_url + "/v1/model/load",
                headers={"X-admin-key": args.admin_key},
                json=request,
                timeout=timeout,
            ) as response:
                gr.Info(f"Load request status: {response.status}")
                if response.status != 200:
                    error_content = await response.text()
                    gr.Error(
                        f"Attempt {attempt_number}: Server returned status {response.status}. Server response: {error_content}"
                    )
                    return False, f"Server error {response.status}: {error_content}"

                async for chunk in response.content:
                    if not model_load_state:
                        raise asyncio.CancelledError("Model load canceled.")
                    chunk_str = chunk.decode("utf-8")
                    if chunk_str.startswith("data: "):
                        data = json.loads(chunk_str.lstrip("data: "))
                        gr.Info(f"Received data: {data}")
                        if data.get("status") == "finished":
                            gr.Info(
                                f"Attempt {attempt_number}: Model loaded successfully"
                            )
                            return True, "Success"
                        elif data.get("status") == "error":
                            error_msg = data.get("message", "")
                            if "Model has max_batch_size * max_input_len" in error_msg:
                                match = re.search(
                                    r"generator requires max_batch_size \* max_q_size = (\d+) \* (\d+) tokens",
                                    error_msg,
                                )
                                if match:
                                    required_tokens = int(match.group(1)) * int(
                                        match.group(2)
                                    )
                                    new_seq_len = ((required_tokens + 255) // 256) * 256
                                    max_seq_len = new_seq_len
                                    cache_size = max(c_size, 2 * max_seq_len)
                                    gr.Info(
                                        f"Attempt {attempt_number}: Adjusting parameters: max_seq_len = {max_seq_len}, cache_size = {cache_size}"
                                    )
                                    return False, f"Error loading model: {error_msg}"
        except asyncio.CancelledError:
            await session.post(
                url=conn_url + "/v1/model/unload",
                headers={"X-admin-key": args.admin_key},
            )
            raise
        except aiohttp.ClientConnectorError as e:
            gr.Error(f"Attempt {attempt_number}: Connection error: {str(e)}")
            return False, f"Connection error: {str(e)}"
        except aiohttp.ClientError as e:
            gr.Error(f"Attempt {attempt_number}: Client error: {str(e)}")
            return False, f"Client error: {str(e)}"
        except Exception as e:
            gr.Error(
                f"Attempt {attempt_number}: Unexpected error during model load: {str(e)}"
            )
            return False, f"Unexpected error: {str(e)}"

    try:
        timeout = aiohttp.ClientTimeout(total=900)  # 15 minutes total timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Unload existing model
            unload_response = await session.post(
                url=conn_url + "/v1/model/unload",
                headers={"X-admin-key": args.admin_key},
            )
            gr.Info(f"Unload request status: {unload_response.status}")
            if unload_response.status != 200:
                unload_content = await unload_response.text()
                gr.Warning(
                    f"Unload request returned status {unload_response.status}: {unload_content}"
                )

            gr.Info(f"Loading {model_name}.")

            success = False
            retry_count = 0
            error_messages = []
            while not success and retry_count < 3:
                retry_count += 1
                success, message = await attempt_load(
                    session, max_seq_len, cache_size, retry_count
                )
                if not success:
                    error_messages.append(f"Attempt {retry_count}: {message}")
                    gr.Info(f"Retrying with new parameters (attempt {retry_count}/3)")

            if success:
                # Verify model load
                gr.Info("Verifying model load...")
                verify_response = await session.get(
                    url=conn_url + "/v1/model", headers={"X-api-key": args.admin_key}
                )
                gr.Info(f"Verification request status: {verify_response.status}")
                if verify_response.status == 200:
                    model_info = await verify_response.json()
                    gr.Info(f"Verification response: {model_info}")
                    if model_info.get("id") == model_name:
                        gr.Info("Model successfully loaded and verified.")
                        return get_current_model(), get_current_loras()
                    else:
                        raise gr.Error(
                            f"Model load reported success, but verification failed. Server reports: {model_info}"
                        )
                elif verify_response.status == 401:
                    verify_content = await verify_response.text()
                    raise gr.Error(
                        f"Authentication failed during model verification. Server response: {verify_content}"
                    )
                else:
                    verify_content = await verify_response.text()
                    raise gr.Error(
                        f"Model load reported success, but verification failed. Server returned status {verify_response.status}: {verify_content}"
                    )
            else:
                error_summary = "\n".join(error_messages)
                raise gr.Error(
                    f"Failed to load the model after multiple attempts:\n{error_summary}"
                )

    except asyncio.CancelledError:
        gr.Info("Model load canceled.")
    except aiohttp.ClientError as e:
        raise gr.Error(f"Client error: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Unexpected error: {str(e)}")
    finally:
        model_load_task = None
        model_load_state = False


def load_loras(loras, scalings):
    if not loras:
        raise gr.Error("Specify at least one lora to load!")
    load_list = []
    for index, lora in enumerate(loras):
        try:
            scaling = float(scalings[0][index])
            load_list.append({"name": lora, "scaling": scaling})
        except ValueError:
            raise gr.Error("Check your scaling values and ensure they are valid!")
    request = {"loras": load_list}
    try:
        requests.post(
            url=conn_url + "/v1/lora/unload", headers={"X-admin-key": args.admin_key}
        )
        r = requests.post(
            url=conn_url + "/v1/lora/load",
            headers={"X-admin-key": args.admin_key},
            json=request,
        )
        r.raise_for_status()
        gr.Info("Loras successfully loaded.")
        return get_current_model(), get_current_loras()
    except Exception as e:
        raise gr.Error(e)


# The unload_model function remains the same as in the previous artifact
def unload_model():
    global model_load_task
    global model_load_state
    if model_load_task and not model_load_task.done():
        model_load_task.cancel()
        model_load_state = False
    else:
        requests.post(
            url=conn_url + "/v1/model/unload", headers={"X-admin-key": args.admin_key}
        )
        gr.Info("Model unloaded.")
    return get_current_model(), get_current_loras()


def unload_loras():
    try:
        r = requests.post(
            url=conn_url + "/v1/lora/unload", headers={"X-admin-key": args.admin_key}
        )
        r.raise_for_status()
        gr.Info("All loras unloaded.")
        return get_current_model(), get_current_loras()
    except Exception as e:
        raise gr.Error(e)


def toggle_gpu_split(gpu_split_auto):
    if gpu_split_auto:
        return gr.Textbox(value=None, visible=False), gr.Textbox(visible=True)
    else:
        return gr.Textbox(visible=True), gr.Textbox(value=None, visible=False)


def load_template(prompt_template):
    try:
        r = requests.post(
            url=conn_url + "/v1/template/switch",
            headers={"X-admin-key": args.admin_key},
            json={"name": prompt_template},
        )
        r.raise_for_status()
        gr.Info(f"Prompt template switched to {prompt_template}.")
        return
    except Exception as e:
        raise gr.Error(e)


def unload_template():
    try:
        r = requests.post(
            url=conn_url + "/v1/template/unload",
            headers={"X-admin-key": args.admin_key},
        )
        r.raise_for_status()
        gr.Info("Prompt template unloaded.")
        return
    except Exception as e:
        raise gr.Error(e)


def load_override(sampler_override):
    try:
        r = requests.post(
            url=conn_url + "/v1/sampling/override/switch",
            headers={"X-admin-key": args.admin_key},
            json={"preset": sampler_override},
        )
        r.raise_for_status()
        gr.Info(f"Sampler override switched to {sampler_override}.")
        return
    except Exception as e:
        raise gr.Error(e)


def unload_override():
    try:
        r = requests.post(
            url=conn_url + "/v1/sampling/override/unload",
            headers={"X-admin-key": args.admin_key},
        )
        r.raise_for_status()
        gr.Info("Sampler override unloaded.")
        return
    except Exception as e:
        raise gr.Error(e)


async def download(repo_id, revision, repo_type, folder_name, token, include, exclude):
    global download_task
    if not folder_name:
        folder_name = repo_id.replace("/", "_")
    include_parsed = ["*"]
    if include:
        include_parsed = [i.strip() for i in list(include.split(","))]
    exclude_parsed = []
    if exclude:
        exclude_parsed = [i.strip() for i in list(include.split(","))]
    request = {
        "repo_id": repo_id,
        "revision": revision,
        "repo_type": repo_type.lower(),
        "folder_name": folder_name,
        "token": token,
        "include": include_parsed,
        "exclude": exclude_parsed,
    }
    try:
        async with aiohttp.ClientSession() as session:
            gr.Info(f"Beginning download of {repo_id}.")
            download_task = asyncio.create_task(
                session.post(
                    url=conn_url + "/v1/download",
                    headers={"X-admin-key": args.admin_key},
                    json=request,
                )
            )
            r = await download_task
            r.raise_for_status()
            content = await r.json()
            gr.Info(
                f'{repo_type} {repo_id} downloaded to folder: {content.get("download_path")}.'
            )
    except asyncio.CancelledError:
        gr.Info("Download canceled.")
    except Exception as e:
        raise gr.Error(e)
    finally:
        await session.close()
        download_task = None


def cancel_download():
    global download_task
    if download_task:
        download_task.cancel()


# Auto-attempt connection if admin key is provided
init_model_text = None
init_lora_text = None
if args.admin_key or args.noauth:
    print("Attempting auto-connection...")
    try:
        models, draft_models, loras, templates, overrides = connect(
            api_url=args.endpoint_url, admin_key=args.admin_key, silent=True
        )
        args.admin_key = args.admin_key if args.admin_key else "noauth"
        init_model_text = get_current_model().value
        init_lora_text = get_current_loras().value
        print(f"Connected models: {models}")
        print(f"Connected draft models: {draft_models}")
        print(f"Connected loras: {loras}")
    except Exception as e:
        print(f"Automatic connection failed: {str(e)}")
        print("Continuing to WebUI.")


# Setup UI elements
with gr.Blocks(title="TabbyAPI Gradio Loader", analytics_enabled=False) as webui:
    gr.Markdown(
        f"## TabbyAPI Exllamav2 Server - Connected to {conn_url}"
        if args.admin_key
        else "## TabbyAPI Exllamav2 Server - Not connected"
    )

    with gr.Row(variant="compact"):
        current_model = gr.Textbox(
            value=init_model_text, label="Current Model:", visible=True, scale=2
        )
        current_loras = gr.Textbox(
            value=init_lora_text, label="Current Loras:", visible=True, scale=1
        )

    with gr.Tabs():
        with gr.Tab("Load Model"):
            with gr.Row(variant="compact"):
                with gr.Column(variant="compact"):
                    with gr.Row(variant="compact"):
                        load_preset = gr.Dropdown(
                            choices=[""] + get_preset_list(True),
                            label="Load Preset:",
                            interactive=True,
                        )
                    with gr.Row(variant="compact"):
                        load_preset_btn = gr.Button(
                            value="📋 Load Preset", variant="primary"
                        )
                        del_preset_btn = gr.Button(
                            value="❌ Delete Preset", variant="stop"
                        )
                with gr.Column(variant="compact"):
                    with gr.Row(variant="compact"):
                        save_preset = gr.Textbox(label="Save Preset:", interactive=True)
                    with gr.Row(variant="compact"):
                        save_preset_btn = gr.Button(
                            value="💾 Save Preset", variant="primary"
                        )
                        refresh_preset_btn = gr.Button(value="♻️ Refresh Presets")
            with gr.Row(variant="compact"):
                with gr.Column(variant="compact"):
                    models_drop = gr.Dropdown(
                        choices=[""] + models, label="Select Model:", interactive=True
                    )
                    draft_models_drop = gr.Dropdown(
                        choices=[""] + draft_models,
                        label="Select Draft Model:",
                        interactive=True,
                        info="Must share the same tokenizer and vocabulary as the primary model.",
                    )
                    prompt_template = gr.Dropdown(
                        choices=[""] + templates,
                        value="",
                        label="Prompt Template:",
                        allow_custom_value=True,
                        interactive=True,
                        info="Jinja2 prompt template to be used for the chat completions endpoint.",
                    )
                with gr.Row(variant="compact"):
                    with gr.Column(variant="compact"):
                        load_model_btn = gr.Button(
                            value="🚀 Load Model", variant="primary"
                        )
                        load_model_btn.scale = "2"
                        unload_model_btn = gr.Button(
                            value="🛑 Cancel Load/Unload Model", variant="stop"
                        )
                        unload_model_btn.scale = "2"
                        with gr.Row(variant="compact", equal_height=True):
                            load_template_btn = gr.Button(
                                value="🚀 Load Template", variant="primary"
                            )
                            unload_template_btn = gr.Button(
                                value="❌ Unload Template", variant="stop"
                            )

            with gr.Group():
                with gr.Row(variant="compact", equal_height=True):
                    with gr.Column(variant="compact"):
                        max_seq_len = gr.Slider(
                            value=8192,
                            label="Max Sequence Length:",
                            minimum=256,
                            maximum=262144,
                            interactive=True,
                            step=256,
                            info="Configured context length to load the model with. If left blank, automatically reads from model config.",
                        )
                        gpu_split_auto = gr.Checkbox(
                            value=True,
                            label="GPU Split Auto",
                            interactive=True,
                            info="Automatically determine how to split model layers between multiple GPUs.",
                        )
                        fasttensors = gr.Checkbox(
                            value=True,
                            label="Use Fasttensors",
                            interactive=True,
                            info="Enable to possibly increase model loading speeds on some systems.",
                        )
                        cache_mode = gr.Radio(
                            value="Q4",
                            label="Cache Mode:",
                            choices=["Q4", "Q6", "Q8", "FP16"],
                            interactive=True,
                            info="Q4/Q6/Q8 cache sacrifice some precision to save VRAM compared to full FP16 precision.",
                        )
                        draft_cache_mode = gr.Radio(
                            value="FP16",
                            label="Draft Cache Mode:",
                            choices=["Q4", "Q6", "Q8", "FP16"],
                            interactive=True,
                            info="Q4/Q6/Q8 cache sacrifice some precision to save VRAM compared to full FP16 precision.",
                        )
                    with gr.Column(variant="compact"):
                        with gr.Row(variant="compact", equal_height=True):
                            cache_size = gr.Slider(
                                label="Cache Size:",
                                value=lambda: None,
                                maximum=262144,
                                step=256,
                                interactive=True,
                                info="Size of the prompt cache to allocate (in number of tokens, multiple of 256). Defaults to max sequence length if left blank.",
                            )
                            chunk_size = gr.Slider(
                                label="Chunk Size:",
                                interactive=True,
                                maximum=32768,
                                minimum=1,
                                info="The number of prompt tokens to ingest at a time. A lower value reduces VRAM usage at the cost of ingestion speed.",
                            )
                        with gr.Row(variant="compact", equal_height=True):
                            autosplit_reserve = gr.Slider(
                                label="Auto-split Reserve:",
                                minimum=0,
                                maximum=1000,
                                value=96,
                                step=8,
                                interactive=True,
                                info="Amount of VRAM to keep reserved on each GPU when using auto split. In megabytes.",
                            )
                            gpu_split = gr.Textbox(
                                label="GPU Split:",
                                placeholder="20.6,24",
                                visible=True,
                                interactive=True,
                                info="Amount of VRAM TabbyAPI will be allowed to use on each GPU. List of numbers separated by commas, in gigabytes.",
                            )

            with gr.Row(variant="compact", equal_height=True):
                with gr.Column(variant="compact"):
                    with gr.Column(variant="compact"):
                        model_rope_scale = gr.Slider(
                            value=lambda: None,
                            label="Rope Scale:",
                            minimum=1,
                            interactive=True,
                            info="AKA compress_pos_emb or linear rope, used for models trained with modified positional embeddings, such as SuperHoT. If left blank, automatically reads from model config.",
                        )
                        model_rope_alpha = gr.Slider(
                            value=lambda: None,
                            label="Rope Alpha:",
                            minimum=1,
                            interactive=True,
                            info="Factor used for NTK-aware rope scaling. Leave blank for automatic calculation based on your configured max_seq_len and the model's base context length.",
                        )
                        num_experts_per_token = gr.Number(
                            value=lambda: None,
                            label="Number of experts per token (MoE only):",
                            precision=0,
                            interactive=True,
                            info="Number of experts to use for simultaneous inference in mixture of experts. If left blank, automatically reads from model config.",
                        )
                with gr.Column(variant="compact"):
                    draft_rope_scale = gr.Number(
                        value=lambda: None,
                        label="Draft Rope Scale:",
                        interactive=True,
                        info="AKA compress_pos_emb or linear rope, used for models trained with modified positional embeddings, such as SuperHoT. If left blank, automatically reads from model config.",
                    )
                    draft_rope_alpha = gr.Number(
                        value=lambda: None,
                        label="Draft Rope Alpha:",
                        interactive=True,
                        info="Factor used for NTK-aware rope scaling. Leave blank for automatic scaling calculated based on your configured max_seq_len and the model's base context length.",
                    )

        with gr.Tab("Sampler Overrides"):
            with gr.Row(variant="compact"):
                with gr.Row(variant="compact"):
                    override_base_seq_len = gr.Slider(
                        value=lambda: None,
                        label="Override Base Sequence Length:",
                        minimum=1,
                        interactive=True,
                        info="Override the model's 'base' sequence length in config.json. Only relevant when using automatic rope alpha. Leave blank if unsure.",
                    )
                    sampler_override = gr.Dropdown(
                        choices=[""] + overrides,
                        value="",
                        label="Select Sampler Overrides:",
                        interactive=True,
                        info="Select a sampler override preset to load.",
                    )
                with gr.Row(variant="compact"):
                    load_override_btn = gr.Button(
                        value="⬆️ Load Override", variant="primary"
                    )
                    unload_override_btn = gr.Button(
                        value="⬇️ Unload Override", variant="stop"
                    )

        with gr.Tab("Load Loras"):
            with gr.Row(variant="compact"):
                load_loras_btn = gr.Button(value="🚀 Load Loras", variant="primary")
                loras_drop = gr.Dropdown(
                    choices=loras,
                    label="Select Loras:",
                    multiselect=True,
                    interactive=True,
                )
                unload_loras_btn = gr.Button(
                    value="❌ Unload All Loras", variant="stop"
                )

            loras_drop = gr.Dropdown(
                label="Select Loras:",
                choices=loras,
                multiselect=True,
                interactive=True,
                info="Select one or more loras to load, specify individual lora weights in the box that appears below (default 1.0).",
            )
            loras_table = gr.List(
                label="Lora Scaling:",
                visible=False,
                datatype="number",
                type="array",
                interactive=True,
            )

        with gr.Tab("HF Downloader"):
            with gr.Row(variant="compact"):
                download_btn = gr.Button(value="Download", variant="primary")
                repo_id = gr.Textbox(label="Repo ID:", interactive=True)
                cancel_download_btn = gr.Button(value="Cancel", variant="stop")

            with gr.Group():
                with gr.Row(variant="compact"):
                    repo_id = gr.Textbox(
                        label="Repo ID:",
                        interactive=True,
                        info="Provided in the format <user/organization name>/<repo name>.",
                    )
                    revision = gr.Textbox(
                        label="Revision/Branch:",
                        interactive=True,
                        info="Name of the revision/branch of the repository to download.",
                    )

                with gr.Row(variant="compact"):
                    repo_type = gr.Dropdown(
                        choices=["Model", "Lora"],
                        value="Model",
                        label="Repo Type:",
                        interactive=True,
                        info="Specify whether the repository contains a model or lora.",
                    )
                    folder_name = gr.Textbox(
                        label="Folder Name:",
                        interactive=True,
                        info="Name to use for the local downloaded copy of the repository.",
                    )

                with gr.Row(variant="compact"):
                    include = gr.Textbox(
                        placeholder="adapter_config.json, adapter_model.bin",
                        label="Include Patterns:",
                        interactive=True,
                        info="Comma-separated list of file patterns to download from repository (default all).",
                    )
                    exclude = gr.Textbox(
                        placeholder="*.bin, *.pth",
                        label="Exclude Patterns:",
                        interactive=True,
                        info="Comma-separated list of file patterns to exclude from download.",
                    )
                with gr.Row(variant="compact"):
                    token = gr.Textbox(
                        label="HF Access Token:",
                        type="password",
                        info="Provide HF access token to download from private/gated repositories.",
                    )

        # Model tab
        load_preset_btn.click(
            fn=read_preset,
            inputs=load_preset,
            outputs=[
                models_drop,
                max_seq_len,
                override_base_seq_len,
                cache_size,
                gpu_split_auto,
                gpu_split,
                model_rope_scale,
                model_rope_alpha,
                cache_mode,
                prompt_template,
                num_experts_per_token,
                draft_models_drop,
                draft_rope_scale,
                draft_rope_alpha,
                draft_cache_mode,
                fasttensors,
                autosplit_reserve,
                chunk_size,
            ],
        )
        del_preset_btn.click(fn=del_preset, inputs=load_preset, outputs=load_preset)
        save_preset_btn.click(
            fn=write_preset,
            inputs=[
                save_preset,
                models_drop,
                max_seq_len,
                override_base_seq_len,
                cache_size,
                gpu_split_auto,
                gpu_split,
                model_rope_scale,
                model_rope_alpha,
                cache_mode,
                prompt_template,
                num_experts_per_token,
                draft_models_drop,
                draft_rope_scale,
                draft_rope_alpha,
                draft_cache_mode,
                fasttensors,
                autosplit_reserve,
                chunk_size,
            ],
            outputs=[save_preset, load_preset],
        )
        refresh_preset_btn.click(fn=get_preset_list, outputs=load_preset)

        gpu_split_auto.change(
            fn=toggle_gpu_split,
            inputs=gpu_split_auto,
            outputs=[gpu_split, autosplit_reserve],
        )
        unload_model_btn.click(fn=unload_model, outputs=[current_model, current_loras])
        load_model_btn.click(
            fn=load_model,
            inputs=[
                models_drop,
                max_seq_len,
                override_base_seq_len,
                cache_size,
                gpu_split_auto,
                gpu_split,
                model_rope_scale,
                model_rope_alpha,
                cache_mode,
                prompt_template,
                num_experts_per_token,
                draft_models_drop,
                draft_rope_scale,
                draft_rope_alpha,
                draft_cache_mode,
                fasttensors,
                autosplit_reserve,
                chunk_size,
            ],
            outputs=[current_model, current_loras],
            concurrency_limit=1,
        )
        load_template_btn.click(fn=load_template, inputs=prompt_template)
        unload_template_btn.click(fn=unload_template)
        load_override_btn.click(fn=load_override, inputs=sampler_override)
        unload_override_btn.click(fn=unload_override)

        # Loras tab
        loras_drop.change(update_loras_table, inputs=loras_drop, outputs=loras_table)
        unload_loras_btn.click(fn=unload_loras, outputs=[current_model, current_loras])
        load_loras_btn.click(
            fn=load_loras,
            inputs=[loras_drop, loras_table],
            outputs=[current_model, current_loras],
        )

        # HF Downloader tab
        download_btn.click(
            fn=download,
            inputs=[repo_id, revision, repo_type, folder_name, token, include, exclude],
            concurrency_limit=1,
        )
        cancel_download_btn.click(fn=cancel_download)

webui.launch(
    inbrowser=args.autolaunch,
    show_api=True,
    server_name=host_url,
    server_port=args.port,
    share=args.share,
)
