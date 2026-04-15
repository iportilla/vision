Dall-e Lab

https://openai.com/index/dall-e/

https://github.com/openai/openai-cookbook/blob/main/examples/dalle/Image_generations_edits_and_variations_with_DALL-E.ipynb


# OpenAI Image API — Notebook Guide

> `Image_generations_edits_and_variations_with_DALL-E.ipynb`  
> Updated to use **`gpt-image-1-mini`** — the `dall-e-2` / `dall-e-3` models have been deprecated.

---

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [API Endpoint Overview](#api-endpoint-overview)
4. [Generation](#generation)
5. [Variations](#variations)
6. [Edits (Inpainting)](#edits-inpainting)
7. [Response Format — b64_json](#response-format)
8. [Model Comparison](#model-comparison)
9. [Running the Notebook](#running-the-notebook)

---

## Overview

The notebook demonstrates three image operations available through the OpenAI Images API:

```mermaid
flowchart LR
    API["OpenAI\nImages API"]
    API --> G["🖼️ Generate\nText → Image"]
    API --> V["🔁 Variations\nImage → Similar Images"]
    API --> E["✏️ Edit / Inpaint\nImage + Mask + Prompt → Edited Image"]

    G --> GM["model: gpt-image-1-mini\nreturns: b64_json"]
    V --> VM["Simulated via\nmultiple generate() calls\n(create_variation removed)"]
    E --> EM["model: gpt-image-1-mini\nmask defines edit region\nreturns: b64_json"]
```

---

## Setup

```mermaid
flowchart TD
    A["Install packages\npip install openai pillow python-dotenv requests"]
    --> B["Create .env file\nOPENAI_API_KEY=sk-..."]
    --> C["load_dotenv('.env')\nos.getenv('OPENAI_API_KEY')"]
    --> D["client = OpenAI(api_key=...)"]
    --> E["Create ./images/ directory\nfor saving output PNGs"]
    --> F["✅ Ready to call API"]
```

### `.env` file format

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
```

---

## API Endpoint Overview

```mermaid
flowchart LR
    subgraph Endpoints["OpenAI Images API Endpoints"]
        direction TB
        GEN["POST /images/generations\nclient.images.generate()"]
        EDI["POST /images/edits\nclient.images.edit()"]
        VAR["POST /images/variations\n⛔ REMOVED — not available\non gpt-image-1 models"]
    end

    subgraph Models["Supported Models"]
        M1["gpt-image-1-mini\nFast, lower cost"]
        M2["gpt-image-1\nHighest quality"]
        M3["dall-e-2 / dall-e-3\n⛔ Deprecated"]
    end

    GEN --> M1
    GEN --> M2
    EDI --> M1
    EDI --> M2
    VAR -. "removed" .-> M3
```

---

## Generation

Text prompt → new image.

```mermaid
flowchart LR
    P["📝 prompt\n'A cyberpunk monkey hacker\ndreaming of bananas...'"]
    --> API["client.images.generate\nmodel='gpt-image-1-mini'\nn=1\nsize='1024x1024'"]
    --> R["ImagesResponse\n.data[0].b64_json\n(base64 string)"]
    --> D["base64.b64decode()\n→ raw bytes"]
    --> F["Write to\ngenerated_image.png"]
    --> SHOW["display(Image.open(...))"]
```

### Key parameters

| Parameter | Value | Notes |
|---|---|---|
| `model` | `gpt-image-1-mini` | Replaces `dall-e-3` |
| `n` | `1` | Only `1` supported by `gpt-image-1-mini` |
| `size` | `1024x1024` | Also: `1536x1024`, `1024x1536`, `auto` |
| `quality` | `auto` | `low` / `medium` / `high` / `auto` |
| `response_format` | *(not set)* | Always returns `b64_json` — URL not supported |

---

## Variations

Generate similar-but-different versions of an image.

> ⚠️ `client.images.create_variation()` has been **removed** from the API.  
> The notebook simulates variations by calling `generate()` multiple times with the same prompt — each call produces a unique interpretation.

```mermaid
flowchart TD
    ORIG["Original image\ngenerated_image.png"]
    PROMPT["Same prompt\nused for generation"]

    ORIG -. "inspiration\n(not sent to API)" .-> LOOP
    PROMPT --> LOOP

    subgraph LOOP["Loop — NUM_VARIATIONS = 2"]
        direction LR
        C1["generate()\niteration 1"]
        C2["generate()\niteration 2"]
    end

    C1 --> V1["variation_image_0.png"]
    C2 --> V2["variation_image_1.png"]

    V1 --> DISP["display() all images\noriginal + variations"]
    V2 --> DISP
```

### Retry logic

Each `generate()` call is wrapped in a retry loop with exponential backoff to handle transient network errors:

```mermaid
flowchart LR
    CALL["generate()"]
    --> TRY{"Success?"}
    TRY -- Yes --> SAVE["Save bytes"]
    TRY -- No, attempt < 3 --> WAIT["sleep(2^attempt)\n2s → 4s"]
    WAIT --> CALL
    TRY -- No, attempt = 3 --> RAISE["raise exception"]
```

---

## Edits (Inpainting)

Replace a masked region of an image with AI-generated content guided by a prompt.

### Step 1 — Create the mask

```mermaid
flowchart TD
    subgraph Mask["1024×1024 Mask PNG (RGBA)"]
        direction LR
        TOP["Top half\nRGBA = (0,0,0,1)\n🔒 Opaque — keep as-is"]
        BOT["Bottom half\nRGBA = (0,0,0,0)\n🔓 Transparent — regenerate this"]
    end

    Mask --> SAVE["bottom_half_mask.png"]
```

### Step 2 — Call the edit endpoint

```mermaid
flowchart LR
    IMG["generated_image.png\n(original image)"]
    MASK["bottom_half_mask.png\n(alpha=0 → edit here)"]
    PROMPT["prompt\n(same as generation)"]

    IMG --> API["client.images.edit\nmodel='gpt-image-1-mini'\nn=1, size='1024x1024'"]
    MASK --> API
    PROMPT --> API

    API --> R["ImagesResponse\n.data[0].b64_json"]
    --> D["base64.b64decode()"]
    --> F["edited_image.png"]
    --> SHOW["display original\n+ edited side by side"]
```

### How the mask works

```mermaid
flowchart LR
    subgraph Before["Input"]
        IM["Full image\n(cyberpunk monkey)"]
        MK["Mask\ntop=opaque\nbottom=transparent"]
    end

    subgraph After["Output"]
        TOP2["Top half\nunchanged 🔒"]
        BOT2["Bottom half\nAI regenerated ✨\nguided by prompt"]
    end

    Before --> After
```

---

## Response Format

`gpt-image-1-mini` **always returns base64-encoded PNG data** — the `response_format="url"` option is not supported.

```mermaid
flowchart LR
    subgraph DALLE["Old: dall-e-3 (deprecated)"]
        direction TB
        A1["response_format='url'"]
        --> B1["response.data[0].url\nhttps://oaidalleapiprodscus.blob.core..."]
        --> C1["requests.get(url).content\n→ bytes"]
    end

    subgraph GPT["New: gpt-image-1-mini"]
        direction TB
        A2["(no response_format needed)"]
        --> B2["response.data[0].b64_json\n'iVBORw0KGgoAAAAN...' (base64 string)"]
        --> C2["base64.b64decode(b64_json)\n→ bytes"]
    end

    C1 --> SAVE["open(filepath,'wb').write(bytes)"]
    C2 --> SAVE
```

---

## Model Comparison

```mermaid
flowchart TD
    subgraph OLD["⛔ Deprecated"]
        DE2["dall-e-2\n• create_variation ✅\n• create_edit ✅\n• size: 256/512/1024\n• response: url or b64"]
        DE3["dall-e-3\n• generate only\n• size: 1024, 1792×1024\n• style: vivid / natural\n• response: url or b64"]
    end

    subgraph NEW["✅ Current"]
        GIM["gpt-image-1-mini\n• generate ✅\n• edit / inpaint ✅\n• variations ⛔ (removed)\n• response: b64_json only\n• n=1 only\n• size: 1024², 1536×1024, auto"]
        GI["gpt-image-1\n• same as mini\n• higher quality\n• higher cost"]
    end

    OLD -. "replaced by" .-> NEW
```

---

## Running the Notebook

```bash
# 1. Install dependencies
pip install openai pillow python-dotenv requests

# 2. Create .env in the same folder as the notebook
echo 'OPENAI_API_KEY=sk-proj-your-key-here' > .env

# 3. Open the notebook
jupyter notebook "Image_generations_edits_and_variations_with_DALL-E.ipynb"
```

### Cell execution order

```mermaid
flowchart TD
    C1["Cell 4 — pip install openai"]
    --> C2["Cell 5 — load .env, get API key"]
    --> C3["Cell 6 — import libraries"]
    --> C4["Cell 7 — init OpenAI client"]
    --> C5["Cell 8 — create images/ directory"]
    --> C6["Cell 10 — generate image"]
    --> C7["Cell 11 — save generated_image.png"]
    --> C8["Cell 12 — display original"]
    --> C9["Cell 14 — generate 2 variations"]
    --> C10["Cell 15 — save variation_image_0/1.png"]
    --> C11["Cell 16 — display all variations"]
    --> C12["Cell 19 — create mask PNG"]
    --> C13["Cell 21 — edit image with mask"]
    --> C14["Cell 22 — save edited_image.png"]
    --> C15["Cell 23 — display original + edited"]
```
