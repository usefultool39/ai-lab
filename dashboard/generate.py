import os, json, re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DASHBOARD = Path(__file__).parent

CATEGORY_ORDER = [
    "AI哲学与总论", "传统机器学习", "概率模型",
    "搜索与早期控制", "深度学习基础", "计算机视觉2D",
    "几何3D视觉", "NLP基础", "Transformer与LLM",
    "生成模型", "多模态", "强化学习",
    "机器人基础", "机器人学习", "Capstone"
]


def parse_task_name(folder_name):
    parts = folder_name.split("-", 2)
    if len(parts) >= 3:
        return {"id": parts[0], "name": parts[1], "en": parts[2]}
    elif len(parts) == 2:
        return {"id": parts[0], "name": parts[1], "en": ""}
    return {"id": folder_name, "name": folder_name, "en": ""}


def parse_readme(readme_path):
    if not readme_path.exists():
        return None
    content = readme_path.read_text(encoding="utf-8")
    data = {"content": content}
    for key, pat in [("status", r"状态[：:]\s*(\S+)"), ("level", r"层级[：:]\s*(\S+)")]:
        m = re.search(pat, content)
        if m:
            data[key] = m.group(1)
    return data


SKIP_DIR_PARTS = {".ipynb_checkpoints", "__pycache__", ".git", ".venv", "venv", "node_modules"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pyd"}


def get_task_files(task_path):
    files = []
    for f in task_path.rglob("*"):
        if not f.is_file():
            continue
        parts = set(f.relative_to(task_path).parts)
        if parts & SKIP_DIR_PARTS:
            continue
        if f.suffix.lower() in SKIP_SUFFIXES:
            continue
        rel = f.relative_to(task_path)
        content = None
        if f.stat().st_size < 100 * 1024:  # <100KB
            try:
                content = f.read_text(encoding="utf-8")
            except Exception:
                pass
        files.append({
            "name": str(rel).replace("\\", "/"),
            "type": f.suffix[1:].lstrip("."),
            "size": f.stat().st_size,
            "content": content
        })
    return files


def scan():
    categories = []
    total = completed = 0

    for cat_folder in sorted(ROOT.iterdir()):
        if not cat_folder.is_dir() or cat_folder.name.startswith(("_", ".")):
            continue
        m = re.match(r"(\d+)-([^(]+)\((\d+)\)", cat_folder.name)
        if not m:
            continue
        cat_id, cat_name, exp = m.groups()
        tasks = []

        for tf in sorted(cat_folder.iterdir()):
            if not tf.is_dir():
                continue
            info = parse_task_name(tf.name)
            info["path"] = str(tf.relative_to(ROOT)).replace("\\", "/")
            info["files"] = get_task_files(tf)

            rd = parse_readme(tf / "README.md")
            if rd:
                info.update(rd)
                if info.get("status") == "完成":
                    completed += 1
            info["has_notes"] = (tf / "学习文档").exists()
            info["has_figures"] = (tf / "figures").exists()
            tasks.append(info)
            total += 1

        categories.append({
            "id": cat_id, "name": cat_name,
            "expected_count": int(exp), "actual_count": len(tasks),
            "tasks": tasks
        })

    categories.sort(key=lambda c: next(
        (i for i, n in enumerate(CATEGORY_ORDER) if n in c["name"]), 999))
    return {
        "categories": categories,
        "stats": {
            "total": total, "completed": completed,
            "progress": round(completed / total * 100, 1) if total else 0
        }
    }


if __name__ == "__main__":
    data = scan()
    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    data_json_compact = json.dumps(data, ensure_ascii=False)

    print("done: {} tasks, {} completed".format(
        data['stats']['total'], data['stats']['completed']))
    print("generating SPA...")

    (DASHBOARD / "data.json").write_text(data_json, encoding="utf-8")
    js_content = "// Auto-generated. Do not edit.\nwindow.__DATA__ = " + data_json_compact + ";\n"
    (DASHBOARD / "data.js").write_text(js_content, encoding="utf-8")
    print("output: " + str(DASHBOARD / "data.js"))
    print("output: " + str(DASHBOARD / "data.json"))
    print("open:  " + str(DASHBOARD / "index.html"))
