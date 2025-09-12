import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from utils import get_sellside_tags, matching

app = Flask(__name__)
CORS(app)

def bad(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code

@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200

@app.post("/tags")
def tags():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "url" not in data:
            return bad("Missing field: url", 421)
        url = data["url"]
        print(f'\nReceived URL: {url}\n')
        if not isinstance(url, str) or not url.strip():
            return bad("url must be a non-empty string.", 422)
        tags, summary = get_sellside_tags(url)
        if not isinstance(tags, pd.DataFrame):
            return bad("utils.extract must return a DataFrame.", 500)
        return jsonify({"tags": tags.to_dict('list'), "summary": summary}), 200
    except Exception as e:
        return jsonify({"error": f"extract error: {e}"}), 500

@app.post("/scores")
def scores():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Body must be a JSON object."}), 421
        if "tags" not in data or "ebitda" not in data:
            return jsonify({"error": "Missing fields: tags and/or ebitda"}), 422
        tags = data["tags"]
        ebitda = data["ebitda"]
        # Expect tags to be dict[str, list[str]]
        if not isinstance(tags, dict) or not all(
            isinstance(k, str) and isinstance(v, list) and all(isinstance(s, str) for s in v)
            for k, v in tags.items()
        ):
            return jsonify({"error": "tags must be a dict[str, list[str]]."}), 422
        if isinstance(ebitda, (int, float)):
            ebitda = float(ebitda)
        else:
            try:
                ebitda = float(ebitda)
            except Exception:
                return jsonify({"error": "ebitda must be a float."}), 422
        scores, deal_size = matching(tags, ebitda)
        # Normalize output
        out = {
            'Scores': [],
            'Deal Size': 0.0
        }
        for i, item in enumerate(scores):
            if not isinstance(item, dict):
                return jsonify({"error": f"Result at index {i} is not an object."}), 500
            for key, typ in (("Nome", str), ("Site", str), ('Resumo', str), ("Score", (int, float)), ('EBITDA', (int, float)), ('Data do EBITDA', str)):
                if key not in item or not isinstance(item[key], typ):
                    return jsonify({"error": f"Result at index {i} missing or invalid key: {key}"}), 500
            item["Score"] = float(item["Score"])
            out['Scores'].append({
                "Nome": item["Nome"],
                "Site": item["Site"],
                "Resumo": item["Resumo"],
                "Score": item["Score"],
                "EBITDA": float(item["EBITDA"]),
                "Data do EBITDA": item["Data do EBITDA"]
            })
        out['Deal Size'] = deal_size
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"error": f"score error: {e}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("DEBUG", "1") == "1")
