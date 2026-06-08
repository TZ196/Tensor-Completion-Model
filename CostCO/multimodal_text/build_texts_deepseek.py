import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request


def read_env(path):
    env = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def read_text(path):
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, sort_keys=True)


def extract_json(text):
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    starts = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not starts:
        raise ValueError("DeepSeek response did not contain JSON")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise ValueError("DeepSeek response had incomplete JSON")
    return json.loads(text[start:end + 1])


def deepseek_chat(env, system_prompt, user_prompt, temperature=0.0,
                  max_tokens=4096, retries=3):
    api_key = env.get("DEEPSEEK_API_KEY")
    if not api_key or api_key == "replace_with_your_key":
        raise ValueError(
            "Set DEEPSEEK_API_KEY in the env file before calling DeepSeek."
        )

    base_url = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = env.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer %s" % api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            if attempt == retries - 1:
                raise
            wait_seconds = 2 ** attempt
            print("DeepSeek request failed, retrying in %ds: %s" %
                  (wait_seconds, exc))
            time.sleep(wait_seconds)

    raise RuntimeError("DeepSeek request failed")


def build_exogenous_prompt(config_text):
    system_prompt = (
        "You are a precise technical summarizer for satellite network machine "
        "learning experiments. You must use only the provided configuration "
        "text and must not invent missing facts."
    )
    user_prompt = """
Task:
Summarize the provided satellite experiment configuration into exogenous text
segments for a multimodal satellite path-traffic tensor completion model.

Rules:
1. Use only the provided configuration text.
2. Do not infer missing parameters.
3. Do not mention validation or test target traffic values.
4. Do not add external satellite-network facts that are absent from the input.
5. Split the summary into concise factual segments.
6. If a field is absent, write "not specified" rather than guessing.
7. Output valid JSON only. Do not wrap it in markdown fences.

Required JSON format:
{
  "metadata": {
    "source": "deepseek_summary_of_experiment_description",
    "text_generation_mode": "deepseek",
    "num_segments": <integer>
  },
  "segments": [
    {
      "segment_id": "C1_simulation_setup",
      "text": "..."
    }
  ]
}

Recommended segment topics:
C1_simulation_setup
C2_constellation_configuration
C3_topology_generation
C4_routing_policy
C5_link_capacity
C6_traffic_generation
C7_tensor_completion_task

Configuration text:
---
%s
---
""" % config_text
    return system_prompt, user_prompt


def build_endogenous_prompt(stats, endo_source, expected_count=None):
    system_prompt = (
        "You generate leakage-safe endogenous time-slice descriptions for a "
        "satellite path-traffic tensor completion model. You must use only the "
        "structured statistics provided."
    )
    if endo_source != "topo":
        raise ValueError("Unsupported endo source: %s" % endo_source)
    allowed = (
        "Use topology fields only. Do not mention traffic counts, traffic "
        "values, traffic trends, source IDs, destination IDs, training masks, "
        "validation masks, test masks, or random masking."
    )
    source_name = "topology_statistics_only"

    slim_stats = []
    for item in stats:
        base = {
            "time_index": item["time_index"],
            "num_satellites": item["num_satellites"],
            "edge_count_undirected": item["edge_count_undirected"],
            "avg_degree": item["avg_degree"],
            "min_degree": item["min_degree"],
            "max_degree": item["max_degree"],
            "is_connected": item["is_connected"],
            "avg_shortest_path_hops": item["avg_shortest_path_hops"],
            "diameter_hops": item["diameter_hops"],
            "changed_edges_from_prev": item["changed_edges_from_prev"],
        }
        slim_stats.append(base)

    if expected_count is None:
        expected_count = len(slim_stats)

    user_prompt = """
Task:
Generate one endogenous English paragraph for each time slice of a satellite
path-traffic tensor completion experiment.

Rules:
1. %s
2. Do not infer, add, or hallucinate facts.
3. Keep wording stable and concise.
4. Each paragraph should be 45-80 words.
5. Output valid JSON only. Do not wrap it in markdown fences.

Required JSON format:
{
  "metadata": {
    "source": "%s",
    "text_generation_mode": "deepseek",
    "num_time_slices": %d
  },
  "texts": [
    {
      "time_index": 0,
      "endo_mode": "%s",
      "text": "..."
    }
  ]
}

Structured statistics:
%s
""" % (
        allowed,
        source_name,
        expected_count,
        endo_source,
        json.dumps(slim_stats, ensure_ascii=False, indent=2),
    )
    return system_prompt, user_prompt


def generate_exogenous(args, env):
    config_text = read_text(args.config_path)
    system_prompt, user_prompt = build_exogenous_prompt(config_text)
    response = deepseek_chat(
        env,
        system_prompt,
        user_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    result = extract_json(response)
    if "segments" not in result:
        raise ValueError("Exogenous DeepSeek JSON must contain segments")
    result.setdefault("metadata", {})
    result["metadata"].update({
        "source_config_path": args.config_path,
            "generator_model": env.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    })
    write_json(args.exo_output_path, result)
    print("Saved DeepSeek exogenous text:", args.exo_output_path)


def save_raw_response(output_path, prefix, response):
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    raw_path = os.path.join(directory, prefix)
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(response)
    return raw_path


def merge_endogenous_chunks(chunks, endo_source, env, stats_path):
    texts = []
    for chunk in chunks:
        texts.extend(chunk["texts"])
    texts = sorted(texts, key=lambda item: int(item["time_index"]))
    return {
        "metadata": {
            "source": "topology_statistics_only",
            "text_generation_mode": "deepseek",
            "num_time_slices": len(texts),
            "stats_path": stats_path,
            "generator_model": env.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "chunked_generation": True,
        },
        "texts": texts,
    }


def generate_endogenous(args, env):
    stats_data = read_json(args.stats_path)
    stats = stats_data["time_statistics"]
    chunks = []
    chunk_size = max(1, args.endo_chunk_size)
    for start in range(0, len(stats), chunk_size):
        chunk_stats = stats[start:start + chunk_size]
        system_prompt, user_prompt = build_endogenous_prompt(
            chunk_stats,
            args.endo_source,
            expected_count=len(stats),
        )
        response = deepseek_chat(
            env,
            system_prompt,
            user_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        try:
            chunk = extract_json(response)
        except json.JSONDecodeError:
            raw_path = save_raw_response(
                args.endo_output_path,
                "endo_deepseek_bad_response_%03d_%03d.txt" %
                (start, start + len(chunk_stats) - 1),
                response,
            )
            raise ValueError(
                "DeepSeek returned invalid JSON for endogenous time slices "
                "%d-%d. Raw response saved to %s" %
                (start, start + len(chunk_stats) - 1, raw_path)
            )
        if "texts" not in chunk:
            raise ValueError("Endogenous DeepSeek JSON must contain texts")
        chunks.append(chunk)

    result = merge_endogenous_chunks(
        chunks,
        args.endo_source,
        env,
        args.stats_path,
    )
    write_json(args.endo_output_path, result)
    print("Saved DeepSeek endogenous text:", args.endo_output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate endogenous/exogenous text with DeepSeek."
    )
    parser.add_argument("--env-path", default="deepseek.env")
    parser.add_argument(
        "--mode",
        choices=["exo", "endo", "both"],
        default="both",
    )
    parser.add_argument(
        "--config-path",
        default="experiment_description.md",
    )
    parser.add_argument(
        "--stats-path",
        default="text_data/time_stats_topo_only.json",
    )
    parser.add_argument(
        "--endo-source",
        choices=["topo"],
        default="topo",
    )
    parser.add_argument(
        "--exo-output-path",
        default="text_data/exo_text_segments.json",
    )
    parser.add_argument(
        "--endo-output-path",
        default="text_data/endo_texts.json",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--endo-chunk-size",
        type=int,
        default=10,
        help="Generate endogenous text in chunks to keep JSON valid.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = read_env(args.env_path)
    if args.mode in ("exo", "both"):
        generate_exogenous(args, env)
    if args.mode in ("endo", "both"):
        generate_endogenous(args, env)


if __name__ == "__main__":
    main()
