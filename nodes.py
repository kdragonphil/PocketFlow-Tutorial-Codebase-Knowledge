import os
import re
import yaml
from pocketflow import Node, BatchNode
from utils.crawl_github_files import crawl_github_files
from utils.call_llm import call_llm
from utils.crawl_local_files import crawl_local_files


# Helper to get content for specific file indices
def get_content_for_indices(files_data, indices):
    content_map = {}
    for i in indices:
        if 0 <= i < len(files_data):
            path, content = files_data[i]
            content_map[f"{i} # {path}"] = (
                content  # Use index + path as key for context
            )
    return content_map


class FetchRepo(Node):
    def prep(self, shared):
        repo_url = shared.get("repo_url")
        local_dir = shared.get("local_dir")
        project_name = shared.get("project_name")

        if not project_name:
            # Basic name derivation from URL or directory
            if repo_url:
                project_name = repo_url.split("/")[-1].replace(".git", "")
            else:
                project_name = os.path.basename(os.path.abspath(local_dir))
            shared["project_name"] = project_name

        # Get file patterns directly from shared
        include_patterns = shared["include_patterns"]
        exclude_patterns = shared["exclude_patterns"]
        max_file_size = shared["max_file_size"]

        return {
            "repo_url": repo_url,
            "local_dir": local_dir,
            "token": shared.get("github_token"),
            "include_patterns": include_patterns,
            "exclude_patterns": exclude_patterns,
            "max_file_size": max_file_size,
            "use_relative_paths": True,
        }

    def exec(self, prep_res):
        if prep_res["repo_url"]:
            print(f"Crawling repository: {prep_res['repo_url']}...")
            result = crawl_github_files(
                repo_url=prep_res["repo_url"],
                token=prep_res["token"],
                include_patterns=prep_res["include_patterns"],
                exclude_patterns=prep_res["exclude_patterns"],
                max_file_size=prep_res["max_file_size"],
                use_relative_paths=prep_res["use_relative_paths"],
            )
        else:
            print(f"Crawling directory: {prep_res['local_dir']}...")

            result = crawl_local_files(
                directory=prep_res["local_dir"],
                include_patterns=prep_res["include_patterns"],
                exclude_patterns=prep_res["exclude_patterns"],
                max_file_size=prep_res["max_file_size"],
                use_relative_paths=prep_res["use_relative_paths"]
            )

        # Convert dict to list of tuples: [(path, content), ...]
        files_list = list(result.get("files", {}).items())
        if len(files_list) == 0:
            raise (ValueError("Failed to fetch files"))
        print(f"Fetched {len(files_list)} files.")
        return files_list

    def post(self, shared, prep_res, exec_res):
        shared["files"] = exec_res  # List of (path, content) tuples


class IdentifyAbstractions(Node):
    def prep(self, shared):
        files_data = shared["files"]
        project_name = shared["project_name"]  # Get project name
        language = shared.get("language", "english")  # Get language
        use_cache = shared.get("use_cache", True)  # Get use_cache flag, default to True
        max_abstraction_num = shared.get("max_abstraction_num", 30)  # Get max_abstraction_num, default to 30

        # Helper to create context from files, respecting limits (basic example)
        def create_llm_context(files_data):
            context = ""
            file_info = []  # Store tuples of (index, path)
            for i, (path, content) in enumerate(files_data):
                entry = f"--- File Index {i}: {path} ---\n{content}\n\n"
                context += entry
                file_info.append((i, path))

            return context, file_info  # file_info is list of (index, path)

        context, file_info = create_llm_context(files_data)
        # Format file info for the prompt (comment is just a hint for LLM)
        file_listing_for_prompt = "\n".join(
            [f"- {idx} # {path}" for idx, path in file_info]
        )
        return (
            context,
            file_listing_for_prompt,
            len(files_data),
            project_name,
            language,
            use_cache,
            max_abstraction_num,
        )  # Return all parameters

    def exec(self, prep_res):
        (
            context,
            file_listing_for_prompt,
            file_count,
            project_name,
            language,
            use_cache,
            max_abstraction_num,
        ) = prep_res  # Unpack all parameters
        print(f"Identifying abstractions using LLM...")

        # Add language instruction and hints only if not English
        language_instruction = ""
        name_lang_hint = ""
        desc_lang_hint = ""
        if language.lower() != "english":
            language_instruction = f"IMPORTANT: Generate the `name` and `description` for each abstraction in **{language.capitalize()}** language. Do NOT use English for these fields.\n\n"
            # Keep specific hints here as name/description are primary targets
            name_lang_hint = f" (value in {language.capitalize()})"
            desc_lang_hint = f" (value in {language.capitalize()})"

        prompt = f"""
For the project `{project_name}`:

Codebase Context:
{context}

{language_instruction}Analyze the codebase context.
Identify the top 5-{max_abstraction_num} core most important abstractions to help those new to the codebase.

For each abstraction, provide:
1. A concise `name`{name_lang_hint}.
2. A beginner-friendly `description` explaining what it is with a simple analogy, in around 100 words{desc_lang_hint}.
3. A list of relevant `file_indices` (integers) using the format `idx # path/comment`.

List of file indices and paths present in the context:
{file_listing_for_prompt}

Format the output as a YAML list of dictionaries:

```yaml
- name: |
    Query Processing{name_lang_hint}
  description: |
    Explains what the abstraction does.
    It's like a central dispatcher routing requests.{desc_lang_hint}
  file_indices:
    - 0 # path/to/file1.py
    - 3 # path/to/related.py
- name: |
    Query Optimization{name_lang_hint}
  description: |
    Another core concept, similar to a blueprint for objects.{desc_lang_hint}
  file_indices:
    - 5 # path/to/another.js
# ... up to {max_abstraction_num} abstractions
```"""
        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0))  # Use cache only if enabled and not retrying

        # --- Validation ---
        yaml_str = response.strip().split("```yaml")[1].split("```")[0].strip()
        abstractions = yaml.safe_load(yaml_str)

        if not isinstance(abstractions, list):
            raise ValueError("LLM Output is not a list")

        validated_abstractions = []
        for item in abstractions:
            if not isinstance(item, dict) or not all(
                k in item for k in ["name", "description", "file_indices"]
            ):
                raise ValueError(f"Missing keys in abstraction item: {item}")
            if not isinstance(item["name"], str):
                raise ValueError(f"Name is not a string in item: {item}")
            if not isinstance(item["description"], str):
                raise ValueError(f"Description is not a string in item: {item}")
            if not isinstance(item["file_indices"], list):
                raise ValueError(f"file_indices is not a list in item: {item}")

            # Validate indices
            validated_indices = []
            for idx_entry in item["file_indices"]:
                try:
                    if isinstance(idx_entry, int):
                        idx = idx_entry
                    elif isinstance(idx_entry, str) and "#" in idx_entry:
                        idx = int(idx_entry.split("#")[0].strip())
                    else:
                        idx = int(str(idx_entry).strip())

                    if not (0 <= idx < file_count):
                        raise ValueError(
                            f"Invalid file index {idx} found in item {item['name']}. Max index is {file_count - 1}."
                        )
                    validated_indices.append(idx)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Could not parse index from entry: {idx_entry} in item {item['name']}"
                    )

            item["files"] = sorted(list(set(validated_indices)))
            # Store only the required fields
            validated_abstractions.append(
                {
                    "name": item["name"],  # Potentially translated name
                    "description": item[
                        "description"
                    ],  # Potentially translated description
                    "files": item["files"],
                }
            )

        print(f"Identified {len(validated_abstractions)} abstractions.")
        return validated_abstractions

    def post(self, shared, prep_res, exec_res):
        shared["abstractions"] = (
            exec_res  # List of {"name": str, "description": str, "files": [int]}
        )


class AnalyzeRelationships(Node):
    def prep(self, shared):
        abstractions = shared[
            "abstractions"
        ]  # Now contains 'files' list of indices, name/description potentially translated
        files_data = shared["files"]
        project_name = shared["project_name"]  # Get project name
        language = shared.get("language", "english")  # Get language
        use_cache = shared.get("use_cache", True)  # Get use_cache flag, default to True

        # Get the actual number of abstractions directly
        num_abstractions = len(abstractions)

        # Create context with abstraction names, indices, descriptions, and relevant file snippets
        context = "Identified Abstractions:\\n"
        all_relevant_indices = set()
        abstraction_info_for_prompt = []
        for i, abstr in enumerate(abstractions):
            # Use 'files' which contains indices directly
            file_indices_str = ", ".join(map(str, abstr["files"]))
            # Abstraction name and description might be translated already
            info_line = f"- Index {i}: {abstr['name']} (Relevant file indices: [{file_indices_str}])\\n  Description: {abstr['description']}"
            context += info_line + "\\n"
            abstraction_info_for_prompt.append(
                f"{i} # {abstr['name']}"
            )  # Use potentially translated name here too
            all_relevant_indices.update(abstr["files"])

        context += "\\nRelevant File Snippets (Referenced by Index and Path):\\n"
        # Get content for relevant files using helper
        relevant_files_content_map = get_content_for_indices(
            files_data, sorted(list(all_relevant_indices))
        )
        # Format file content for context
        file_context_str = "\\n\\n".join(
            f"--- File: {idx_path} ---\\n{content}"
            for idx_path, content in relevant_files_content_map.items()
        )
        context += file_context_str

        return (
            context,
            "\n".join(abstraction_info_for_prompt),
            num_abstractions, # Pass the actual count
            project_name,
            language,
            use_cache,
        )  # Return use_cache

    def exec(self, prep_res):
        (
            context,
            abstraction_listing,
            num_abstractions, # Receive the actual count
            project_name,
            language,
            use_cache,
         ) = prep_res  # Unpack use_cache
        print(f"Analyzing relationships using LLM...")

        # Add language instruction and hints only if not English
        language_instruction = ""
        lang_hint = ""
        list_lang_note = ""
        if language.lower() != "english":
            language_instruction = f"IMPORTANT: Generate the `summary` and relationship `label` fields in **{language.capitalize()}** language. Do NOT use English for these fields.\n\n"
            lang_hint = f" (in {language.capitalize()})"
            list_lang_note = f" (Names might be in {language.capitalize()})"  # Note for the input list

        prompt = f"""
Based on the following abstractions and relevant code snippets from the project `{project_name}`:

List of Abstraction Indices and Names{list_lang_note}:
{abstraction_listing}

Context (Abstractions, Descriptions, Code):
{context}

{language_instruction}Please provide:
1. A high-level `summary` of the project's main purpose and functionality in a few beginner-friendly sentences{lang_hint}. Use markdown formatting with **bold** and *italic* text to highlight important concepts.
2. A list (`relationships`) describing the key interactions between these abstractions. For each relationship, specify:
    - `from_abstraction`: Index of the source abstraction (e.g., `0 # AbstractionName1`)
    - `to_abstraction`: Index of the target abstraction (e.g., `1 # AbstractionName2`)
    - `label`: A brief label for the interaction **in just a few words**{lang_hint} (e.g., "Manages", "Inherits", "Uses").
    Ideally the relationship should be backed by one abstraction calling or passing parameters to another.
    Simplify the relationship and exclude those non-important ones.

IMPORTANT: Make sure EVERY abstraction is involved in at least ONE relationship (either as source or target). Each abstraction index must appear at least once across all relationships.

Format the output as YAML:

```yaml
summary: |
  A brief, simple explanation of the project{lang_hint}.
  Can span multiple lines with **bold** and *italic* for emphasis.
relationships:
  - from_abstraction: 0 # AbstractionName1
    to_abstraction: 1 # AbstractionName2
    label: "Manages"{lang_hint}
  - from_abstraction: 2 # AbstractionName3
    to_abstraction: 0 # AbstractionName1
    label: "Provides config"{lang_hint}
  # ... other relationships
```

Now, provide the YAML output:
"""
        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0)) # Use cache only if enabled and not retrying

        # --- Validation ---
        yaml_str = response.strip().split("```yaml")[1].split("```")[0].strip()
        relationships_data = yaml.safe_load(yaml_str)

        if not isinstance(relationships_data, dict) or not all(
            k in relationships_data for k in ["summary", "relationships"]
        ):
            raise ValueError(
                "LLM output is not a dict or missing keys ('summary', 'relationships')"
            )
        if not isinstance(relationships_data["summary"], str):
            raise ValueError("summary is not a string")
        if not isinstance(relationships_data["relationships"], list):
            raise ValueError("relationships is not a list")

        # Validate relationships structure
        validated_relationships = []
        for rel in relationships_data["relationships"]:
            # Check for 'label' key
            if not isinstance(rel, dict) or not all(
                k in rel for k in ["from_abstraction", "to_abstraction", "label"]
            ):
                raise ValueError(
                    f"Missing keys (expected from_abstraction, to_abstraction, label) in relationship item: {rel}"
                )
            # Validate 'label' is a string
            if not isinstance(rel["label"], str):
                raise ValueError(f"Relationship label is not a string: {rel}")

            # Validate indices
            try:
                from_idx = int(str(rel["from_abstraction"]).split("#")[0].strip())
                to_idx = int(str(rel["to_abstraction"]).split("#")[0].strip())
                if not (
                    0 <= from_idx < num_abstractions and 0 <= to_idx < num_abstractions
                ):
                    raise ValueError(
                        f"Invalid index in relationship: from={from_idx}, to={to_idx}. Max index is {num_abstractions-1}."
                    )
                validated_relationships.append(
                    {
                        "from": from_idx,
                        "to": to_idx,
                        "label": rel["label"],  # Potentially translated label
                    }
                )
            except (ValueError, TypeError):
                raise ValueError(f"Could not parse indices from relationship: {rel}")

        print("Generated project summary and relationship details.")
        return {
            "summary": relationships_data["summary"],  # Potentially translated summary
            "details": validated_relationships,  # Store validated, index-based relationships with potentially translated labels
        }

    def post(self, shared, prep_res, exec_res):
        # Structure is now {"summary": str, "details": [{"from": int, "to": int, "label": str}]}
        # Summary and label might be translated
        shared["relationships"] = exec_res


class OrderChapters(Node):
    def prep(self, shared):
        abstractions = shared["abstractions"]  # Name/description might be translated
        relationships = shared["relationships"]  # Summary/label might be translated
        project_name = shared["project_name"]  # Get project name
        language = shared.get("language", "english")  # Get language
        use_cache = shared.get("use_cache", True)  # Get use_cache flag, default to True

        # Prepare context for the LLM
        abstraction_info_for_prompt = []
        for i, a in enumerate(abstractions):
            abstraction_info_for_prompt.append(
                f"- {i} # {a['name']}"
            )  # Use potentially translated name
        abstraction_listing = "\n".join(abstraction_info_for_prompt)

        # Use potentially translated summary and labels
        summary_note = ""
        if language.lower() != "english":
            summary_note = (
                f" (Note: Project Summary might be in {language.capitalize()})"
            )

        context = f"Project Summary{summary_note}:\n{relationships['summary']}\n\n"
        context += "Relationships (Indices refer to abstractions above):\n"
        for rel in relationships["details"]:
            from_name = abstractions[rel["from"]]["name"]
            to_name = abstractions[rel["to"]]["name"]
            # Use potentially translated 'label'
            context += f"- From {rel['from']} ({from_name}) to {rel['to']} ({to_name}): {rel['label']}\n"  # Label might be translated

        list_lang_note = ""
        if language.lower() != "english":
            list_lang_note = f" (Names might be in {language.capitalize()})"

        return (
            abstraction_listing,
            context,
            len(abstractions),
            project_name,
            list_lang_note,
            use_cache,
        )  # Return use_cache

    def exec(self, prep_res):
        (
            abstraction_listing,
            context,
            num_abstractions,
            project_name,
            list_lang_note,
            use_cache,
        ) = prep_res  # Unpack use_cache
        print("Determining chapter order using LLM...")
        # No language variation needed here in prompt instructions, just ordering based on structure
        # The input names might be translated, hence the note.
        prompt = f"""
Given the following project abstractions and their relationships for the project ```` {project_name} ````:

Abstractions (Index # Name){list_lang_note}:
{abstraction_listing}

Context about relationships and project summary:
{context}

If you are going to make a tutorial for ```` {project_name} ````, what is the best order to explain these abstractions, from first to last?
Ideally, first explain those that are the most important or foundational, perhaps user-facing concepts or entry points. Then move to more detailed, lower-level implementation details or supporting concepts.

Output the ordered list of abstraction indices, including the name in a comment for clarity. Use the format `idx # AbstractionName`.

```yaml
- 2 # FoundationalConcept
- 0 # CoreClassA
- 1 # CoreClassB (uses CoreClassA)
- ...
```

Now, provide the YAML output:
"""
        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0)) # Use cache only if enabled and not retrying

        # --- Validation ---
        yaml_str = response.strip().split("```yaml")[1].split("```")[0].strip()
        ordered_indices_raw = yaml.safe_load(yaml_str)

        if not isinstance(ordered_indices_raw, list):
            raise ValueError("LLM output is not a list")

        ordered_indices = []
        seen_indices = set()
        for entry in ordered_indices_raw:
            try:
                if isinstance(entry, int):
                    idx = entry
                elif isinstance(entry, str) and "#" in entry:
                    idx = int(entry.split("#")[0].strip())
                else:
                    idx = int(str(entry).strip())

                if not (0 <= idx < num_abstractions):
                    raise ValueError(
                        f"Invalid index {idx} in ordered list. Max index is {num_abstractions-1}."
                    )
                if idx in seen_indices:
                    raise ValueError(f"Duplicate index {idx} found in ordered list.")
                ordered_indices.append(idx)
                seen_indices.add(idx)

            except (ValueError, TypeError):
                raise ValueError(
                    f"Could not parse index from ordered list entry: {entry}"
                )

        # Check if all abstractions are included
        if len(ordered_indices) != num_abstractions:
            raise ValueError(
                f"Ordered list length ({len(ordered_indices)}) does not match number of abstractions ({num_abstractions}). Missing indices: {set(range(num_abstractions)) - seen_indices}"
            )

        print(f"Determined chapter order (indices): {ordered_indices}")
        return ordered_indices  # Return the list of indices

    def post(self, shared, prep_res, exec_res):
        # exec_res is already the list of ordered indices
        shared["chapter_order"] = exec_res  # List of indices


class AnalyzeAPICalls(Node):
    def prep(self, shared):
        files_data = shared["files"]  # List of (path, content) tuples
        project_name = shared["project_name"]
        language = shared.get("language", "english")
        use_cache = shared.get("use_cache", True)

        # Filter for frontend files (JavaScript, TypeScript)
        frontend_files = []
        for path, content in files_data:
            if path.endswith((".js", ".jsx", ".ts", ".tsx")):
                frontend_files.append({"path": path, "content": content})

        if not frontend_files:
            print("No frontend files (JS/TS) found to analyze for API calls.")
            return None # Skip exec if no relevant files

        return {
            "frontend_files": frontend_files,
            "project_name": project_name,
            "language": language,
            "use_cache": use_cache,
        }

    def exec(self, prep_res):
        if prep_res is None:
            return [] # Return empty list if prep returned None

        frontend_files = prep_res["frontend_files"]
        project_name = prep_res["project_name"]
        language = prep_res["language"]
        use_cache = prep_res["use_cache"]

        all_api_calls_info = []
        print(f"Analyzing API calls in {len(frontend_files)} frontend files using LLM...")

        for file_info in frontend_files:
            file_path = file_info["path"]
            file_content = file_info["content"]

            # Add language instruction and hints only if not English
            # While the primary analysis is on code, the output YAML structure might be described or confirmed in the target language.
            language_instruction = ""
            yaml_lang_hint = ""
            if language.lower() != "english":
                language_instruction = f"IMPORTANT: The response should be YAML. If you add any descriptive text outside the YAML, it should be in **{language.capitalize()}** language.\n\n"
                yaml_lang_hint = f" (values for description/notes, if any, should be in {language.capitalize()})"


            prompt = f"""
{language_instruction}For the project `{project_name}`, and the file `{file_path}`:

File Content:
```{'javascript' if file_path.endswith(('.js', '.jsx')) else 'typescript'}
{file_content}
```

Analyze the frontend code (JavaScript/TypeScript) above.
Identify all API calls (e.g., using `fetch`, `axios`, `XMLHttpRequest`, or other HTTP client libraries).

For each API call found, provide the following details:
1.  `calling_function_name`: The name of the function in which the API call is made. If it's not in a function, use "global scope" or a relevant class/method name.
2.  `api_endpoint`: The URL or endpoint of the API being called. If it's a variable, provide the variable name.
3.  `http_method`: The HTTP method used (e.g., GET, POST, PUT, DELETE).
4.  `request_parameters`: A list of key-value pairs or a description of parameters sent with the request (query parameters, request body, headers if significant). {yaml_lang_hint}
5.  `response_usage`: A description of how the API response data is used in the code (e.g., "response.data is stored in `userData` state", "items from response are mapped to UI components"). {yaml_lang_hint}

Format the output as a YAML list of dictionaries, with one dictionary per API call found in this file.
If no API calls are found in this file, output an empty YAML list `[]`.

Example for a single API call:
```yaml
- calling_function_name: "fetchUserDetails"
  api_endpoint: "/api/users/{{userId}}" # or variable name like "API_BASE_URL + '/users/' + userId"
  http_method: "GET"
  request_parameters:
    - name: "userId"
      source: "path_variable" # e.g., path_variable, query_param, request_body, header
      description: "User's unique identifier" {yaml_lang_hint}
  response_usage: "The user's name (response.name) and email (response.email) are displayed in the profile section." {yaml_lang_hint}
```

Now, provide the YAML output for the file `{file_path}`:
"""
            try:
                response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0))
                yaml_str = response.strip()
                if "```yaml" in yaml_str:
                    yaml_str = yaml_str.split("```yaml")[1].split("```")[0].strip()
                elif "```" in yaml_str: # Handle cases where only ``` is present
                    yaml_str = yaml_str.split("```")[1].strip()


                # Ensure it's valid YAML, even if empty
                if not yaml_str:
                    api_calls_in_file = []
                else:
                    api_calls_in_file = yaml.safe_load(yaml_str)

                if not isinstance(api_calls_in_file, list):
                    print(f"Warning: LLM output for {file_path} was not a list, but: {type(api_calls_in_file)}. Treating as no API calls found.")
                    api_calls_in_file = []


                if api_calls_in_file: # Only add if there's content
                    all_api_calls_info.append({
                        "file_path": file_path,
                        "api_calls": api_calls_in_file
                    })
                    print(f"  - Found {len(api_calls_in_file)} API call(s) in {file_path}")

            except yaml.YAMLError as e:
                print(f"Error parsing YAML from LLM response for {file_path}: {e}")
                print(f"LLM Response was:\n{response}")
            except Exception as e:
                print(f"Error processing file {file_path} for API calls: {e}")
                print(f"LLM Response was (if available):\n{response if 'response' in locals() else 'N/A'}")


        if not all_api_calls_info:
            print("No API calls identified in any frontend files.")
        else:
            print(f"Identified API calls in {len(all_api_calls_info)} file(s).")
        return all_api_calls_info

    def post(self, shared, prep_res, exec_res):
        shared["api_call_analysis"] = exec_res


class AnalyzeFastAPIEndpoints(Node):
    def prep(self, shared):
        files_data = shared["files"]  # List of (path, content) tuples
        project_name = shared["project_name"]
        language = shared.get("language", "english") # For potential descriptions in YAML
        use_cache = shared.get("use_cache", True)

        # Filter for Python files
        python_files = []
        for path, content in files_data:
            if path.endswith((".py")):
                python_files.append({"path": path, "content": content})

        if not python_files:
            print("No Python files found to analyze for FastAPI endpoints.")
            return None # Skip exec if no relevant files

        return {
            "python_files": python_files,
            "project_name": project_name,
            "language": language,
            "use_cache": use_cache,
        }

    def exec(self, prep_res):
        if prep_res is None:
            return [] # Return empty list if prep returned None

        python_files = prep_res["python_files"]
        project_name = prep_res["project_name"]
        language = prep_res["language"] # For descriptions in YAML
        use_cache = prep_res["use_cache"]

        all_endpoints_info = []
        print(f"Analyzing FastAPI endpoints in {len(python_files)} Python files using LLM...")

        for file_info in python_files:
            file_path = file_info["path"]
            file_content = file_info["content"]

            language_instruction = ""
            yaml_desc_hint = ""
            if language.lower() != "english":
                language_instruction = f"IMPORTANT: The response MUST be YAML. If you include any descriptive text for fields like 'description', it should be in **{language.capitalize()}** language.\n\n"
                yaml_desc_hint = f" (in {language.capitalize()})"

            prompt = f"""
{language_instruction}For the project `{project_name}`, and the Python file `{file_path}`:

File Content:
```python
{file_content}
```

Analyze the Python code above to identify FastAPI endpoints.
For each FastAPI endpoint (e.g., defined with `@app.get`, `@router.post`, etc.), extract the following information:

1.  `http_method`: The HTTP method (e.g., GET, POST, PUT, DELETE).
2.  `route_path`: The URL path for the endpoint (e.g., "/items/{{item_id}}").
3.  `summary`: A brief summary or description of the endpoint, often found in the function's docstring or comments above it{yaml_desc_hint}.
4.  `path_parameters`: A list of path parameters. For each, include:
    *   `name`: Parameter name (e.g., "item_id").
    *   `type`: Parameter type (e.g., "int", "str"){yaml_desc_hint}.
5.  `query_parameters`: A list of query parameters. For each, include:
    *   `name`: Parameter name (e.g., "limit").
    *   `type`: Parameter type (e.g., "int", "str"){yaml_desc_hint}.
    *   `default` (optional): Default value if specified.
    *   `required` (optional): Boolean, true if the parameter is required, false or absent otherwise.
6.  `request_body_model`: Information about the request body, if any. Include:
    *   `model_name`: The Pydantic model name (e.g., "ItemCreate").
    *   `fields`: A list of fields in the model, each with `name`, `type`{yaml_desc_hint}, and `required` (boolean).
    *   `example` (optional): A simple JSON example of the request body{yaml_desc_hint}.
7.  `response_model`: Information about the response. Include:
    *   `model_name`: The Pydantic model name (e.g., "ItemRead").
    *   `fields`: A list of fields in the model, each with `name` and `type`{yaml_desc_hint}.
    *   `example` (optional): A simple JSON example of the response body{yaml_desc_hint}.
    *   `status_code` (optional): The primary HTTP status code for successful responses (e.g., 200, 201).

Format the output as a YAML list of dictionaries, with one dictionary per endpoint found in this file.
If no FastAPI endpoints are found in this file, output an empty YAML list `[]`.

Example for a single endpoint:
```yaml
- http_method: "POST"
  route_path: "/items/"
  summary: "Create a new item."{yaml_desc_hint}
  path_parameters: []
  query_parameters: []
  request_body_model:
    model_name: "ItemCreate"
    fields:
      - name: "name"
        type: "str"
        required: true
      - name: "price"
        type: "float"
        required: true
      - name: "description"
        type: "Optional[str]"
        required: false
    example:
      name: "My Item"
      price: 10.5
      description: "A cool item."
  response_model:
    model_name: "ItemRead"
    fields:
      - name: "id"
        type: "int"
      - name: "name"
        type: "str"
      - name: "price"
        type: "float"
      - name: "description"
        type: "Optional[str]"
    status_code: 201
    example:
      id: 1
      name: "My Item"
      price: 10.5
      description: "A cool item."
```

Now, provide the YAML output for the file `{file_path}`:
"""
            try:
                response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0))
                yaml_str = response.strip()
                if "```yaml" in yaml_str:
                    yaml_str = yaml_str.split("```yaml")[1].split("```")[0].strip()
                elif "```" in yaml_str:
                    yaml_str = yaml_str.split("```")[1].strip()

                if not yaml_str:
                    endpoints_in_file = []
                else:
                    endpoints_in_file = yaml.safe_load(yaml_str)

                if not isinstance(endpoints_in_file, list):
                    print(f"Warning: LLM output for {file_path} (FastAPI) was not a list, but: {type(endpoints_in_file)}. Treating as no endpoints found.")
                    endpoints_in_file = []

                if endpoints_in_file:
                    all_endpoints_info.append({
                        "file_path": file_path,
                        "endpoints": endpoints_in_file
                    })
                    print(f"  - Found {len(endpoints_in_file)} FastAPI endpoint(s) in {file_path}")

            except yaml.YAMLError as e:
                print(f"Error parsing YAML from LLM response for {file_path} (FastAPI): {e}")
                print(f"LLM Response was:\n{response}")
            except Exception as e:
                print(f"Error processing file {file_path} for FastAPI endpoints: {e}")
                print(f"LLM Response was (if available):\n{response if 'response' in locals() else 'N/A'}")

        if not all_endpoints_info:
            print("No FastAPI endpoints identified in any Python files.")
        else:
            print(f"Identified FastAPI endpoints in {len(all_endpoints_info)} file(s).")
        return all_endpoints_info

    def post(self, shared, prep_res, exec_res):
        shared["fastapi_endpoint_analysis"] = exec_res


class GenerateAPIDocumentation(Node):
    def prep(self, shared):
        fastapi_analysis_data = shared.get("fastapi_endpoint_analysis", [])
        project_name = shared["project_name"]
        language = shared.get("language", "english")
        use_cache = shared.get("use_cache", True)

        if not fastapi_analysis_data:
            print("No FastAPI endpoint analysis data found to generate API documentation.")
            return None

        return {
            "fastapi_analysis_data": fastapi_analysis_data,
            "project_name": project_name,
            "language": language,
            "use_cache": use_cache,
        }

    def exec(self, prep_res):
        if prep_res is None:
            return "" # Return empty string if no data

        fastapi_analysis_data = prep_res["fastapi_analysis_data"]
        project_name = prep_res["project_name"]
        language = prep_res["language"]
        use_cache = prep_res["use_cache"]

        print(f"Generating API documentation for {project_name} using LLM...")

        # Construct a string representation of the endpoint data for the prompt
        endpoint_data_str_parts = []
        for file_analysis in fastapi_analysis_data:
            file_path = file_analysis.get("file_path", "Unknown file")
            endpoint_data_str_parts.append(f"Endpoints from file: {file_path}\n")
            if isinstance(file_analysis.get("endpoints"), list):
                for endpoint in file_analysis["endpoints"]:
                    endpoint_data_str_parts.append(f"- Method: {endpoint.get('http_method')}, Path: {endpoint.get('route_path')}")
                    endpoint_data_str_parts.append(f"  Summary: {endpoint.get('summary', 'N/A')}")
                    # Add more details as needed for the prompt, e.g., parameters, request/response bodies
                    # This part can be expanded to make the YAML string more complete for the LLM context
                    if endpoint.get("path_parameters"):
                        endpoint_data_str_parts.append(f"  Path Params: {endpoint.get('path_parameters')}")
                    if endpoint.get("query_parameters"):
                        endpoint_data_str_parts.append(f"  Query Params: {endpoint.get('query_parameters')}")
                    if endpoint.get("request_body_model"):
                        endpoint_data_str_parts.append(f"  Request Body: {endpoint.get('request_body_model')}")
                    if endpoint.get("response_model"):
                        endpoint_data_str_parts.append(f"  Response Model: {endpoint.get('response_model')}")
            endpoint_data_str_parts.append("\n")
        full_endpoint_data_for_prompt = "\n".join(endpoint_data_str_parts)

        language_instruction = ""
        doc_lang_note = ""
        if language.lower() != "english":
            lang_cap = language.capitalize()
            language_instruction = f"IMPORTANT: Generate the ENTIRE API documentation in **{lang_cap}**. Input data (summaries, types) might already be in {lang_cap}, but all surrounding text, explanations, and section titles MUST be in {lang_cap}. DO NOT use English except for technical keywords like HTTP methods, or Pydantic model names if they are intrinsically English.\n\n"
            doc_lang_note = f" (Translate all descriptive text to {lang_cap})"

        prompt = f"""
{language_instruction}Project Name: {project_name}

FastAPI Endpoint Data (extracted from source code):
```yaml
{full_endpoint_data_for_prompt}
```

Based on the structured FastAPI endpoint data provided above, generate a comprehensive API documentation in Markdown format, specifically for frontend developers{doc_lang_note}.

The documentation should include:
1.  A main title for the API documentation (e.g., "API Reference for {project_name}"){doc_lang_note}.
2.  An introductory section briefly explaining what the API does or how to use the documentation{doc_lang_note}.
3.  For each endpoint, create a section with:
    *   A clear title including the HTTP method and route path (e.g., `POST /items/`){doc_lang_note}.
    *   The summary/description of the endpoint{doc_lang_note}.
    *   Path Parameters: If any, list them in a table with columns for `Name`, `Type`, and `Description`{doc_lang_note}.
    *   Query Parameters: If any, list them in a table with columns for `Name`, `Type`, `Required`, `Default`, and `Description`{doc_lang_note}.
    *   Request Body: If applicable, describe the expected request body. Include the Pydantic model name, its fields (with `Name`, `Type`, `Required`), and a JSON example{doc_lang_note}.
    *   Response Model: Describe the expected response. Include the Pydantic model name, its fields (with `Name`, `Type`), the success status code, and a JSON example of the response{doc_lang_note}.

Use clear Markdown formatting, including headings, tables, and code blocks for JSON examples.
Ensure the language used is beginner-friendly for frontend developers and all descriptive text is in the target language specified ({language.capitalize() if language.lower() != 'english' else 'English'}).

Output *only* the Markdown content for this API documentation.
Do NOT include ```markdown``` tags around the output.

Begin the documentation now:
"""

        try:
            api_doc_markdown = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0))
            print(f"Successfully generated API documentation for {project_name}.")
            return api_doc_markdown.strip()
        except Exception as e:
            print(f"Error generating API documentation for {project_name}: {e}")
            return "" # Return empty string on error

    def post(self, shared, prep_res, exec_res):
        shared["api_documentation_md"] = exec_res


class WriteChapters(BatchNode):
    def prep(self, shared):
        chapter_order = shared["chapter_order"]  # List of indices
        abstractions = shared[
            "abstractions"
        ]  # List of {"name": str, "description": str, "files": [int]}
        files_data = shared["files"]  # List of (path, content) tuples
        project_name = shared["project_name"]
        language = shared.get("language", "english")
        use_cache = shared.get("use_cache", True)  # Get use_cache flag, default to True
        api_call_analysis = shared.get("api_call_analysis", []) # Get API call analysis

        # Get already written chapters to provide context
        # We store them temporarily during the batch run, not in shared memory yet
        # The 'previous_chapters_summary' will be built progressively in the exec context
        self.chapters_written_so_far = (
            []
        )  # Use instance variable for temporary storage across exec calls

        # Create a complete list of all chapters
        all_chapters = []
        chapter_filenames = {}  # Store chapter filename mapping for linking
        for i, abstraction_index in enumerate(chapter_order):
            if 0 <= abstraction_index < len(abstractions):
                chapter_num = i + 1
                chapter_name = abstractions[abstraction_index][
                    "name"
                ]  # Potentially translated name
                # Create safe filename (from potentially translated name)
                safe_name = "".join(
                    c if c.isalnum() else "_" for c in chapter_name
                ).lower()
                filename = f"{i+1:02d}_{safe_name}.md"
                # Format with link (using potentially translated name)
                all_chapters.append(f"{chapter_num}. [{chapter_name}]({filename})")
                # Store mapping of chapter index to filename for linking
                chapter_filenames[abstraction_index] = {
                    "num": chapter_num,
                    "name": chapter_name,
                    "filename": filename,
                }

        # Create a formatted string with all chapters
        full_chapter_listing = "\n".join(all_chapters)

        items_to_process = []
        for i, abstraction_index in enumerate(chapter_order):
            if 0 <= abstraction_index < len(abstractions):
                abstraction_details = abstractions[
                    abstraction_index
                ]  # Contains potentially translated name/desc
                # Use 'files' (list of indices) directly
                related_file_indices = abstraction_details.get("files", [])
                # Get content using helper, passing indices
                related_files_content_map = get_content_for_indices(
                    files_data, related_file_indices
                )

                # Find relevant API calls for this abstraction's files
                relevant_api_calls_for_abstraction = []
                abstraction_file_paths = [files_data[idx][0] for idx in related_file_indices]

                for analysis_item in api_call_analysis:
                    if analysis_item["file_path"] in abstraction_file_paths:
                        relevant_api_calls_for_abstraction.append(analysis_item)


                # Get previous chapter info for transitions (uses potentially translated name)
                prev_chapter = None
                if i > 0:
                    prev_idx = chapter_order[i - 1]
                    prev_chapter = chapter_filenames[prev_idx]

                # Get next chapter info for transitions (uses potentially translated name)
                next_chapter = None
                if i < len(chapter_order) - 1:
                    next_idx = chapter_order[i + 1]
                    next_chapter = chapter_filenames[next_idx]

                items_to_process.append(
                    {
                        "chapter_num": i + 1,
                        "abstraction_index": abstraction_index,
                        "abstraction_details": abstraction_details,  # Has potentially translated name/desc
                        "related_files_content_map": related_files_content_map,
                        "project_name": shared["project_name"],  # Add project name
                        "full_chapter_listing": full_chapter_listing,  # Add the full chapter listing (uses potentially translated names)
                        "chapter_filenames": chapter_filenames,  # Add chapter filenames mapping (uses potentially translated names)
                        "prev_chapter": prev_chapter,  # Add previous chapter info (uses potentially translated name)
                        "next_chapter": next_chapter,  # Add next chapter info (uses potentially translated name)
                        "language": language,  # Add language for multi-language support
                        "use_cache": use_cache, # Pass use_cache flag
                        "api_calls_for_chapter": relevant_api_calls_for_abstraction, # Add relevant API calls
                        # previous_chapters_summary will be added dynamically in exec
                    }
                )
            else:
                print(
                    f"Warning: Invalid abstraction index {abstraction_index} in chapter_order. Skipping."
                )

        print(f"Preparing to write {len(items_to_process)} chapters...")
        return items_to_process  # Iterable for BatchNode

    def exec(self, item):
        # This runs for each item prepared above
        abstraction_name = item["abstraction_details"][
            "name"
        ]  # Potentially translated name
        abstraction_description = item["abstraction_details"][
            "description"
        ]  # Potentially translated description
        chapter_num = item["chapter_num"]
        project_name = item.get("project_name")
        language = item.get("language", "english")
        use_cache = item.get("use_cache", True) # Read use_cache from item
        api_calls_for_chapter = item.get("api_calls_for_chapter", []) # Get API calls for this chapter

        print(f"Writing chapter {chapter_num} for: {abstraction_name} using LLM...")

        # Prepare file context string from the map
        file_context_str = "\n\n".join(
            f"--- File: {idx_path.split('# ')[1] if '# ' in idx_path else idx_path} ---\n{content}"
            for idx_path, content in item["related_files_content_map"].items()
        )

        # Get summary of chapters written *before* this one
        # Use the temporary instance variable
        previous_chapters_summary = "\n---\n".join(self.chapters_written_so_far)

        # Add language instruction and context notes only if not English
        language_instruction = ""
        concept_details_note = ""
        structure_note = ""
        prev_summary_note = ""
        instruction_lang_note = ""
        mermaid_lang_note = ""
        code_comment_note = ""
        link_lang_note = ""
        tone_note = ""
        api_info_note = ""
        if language.lower() != "english":
            lang_cap = language.capitalize()
            language_instruction = f"IMPORTANT: Write this ENTIRE tutorial chapter in **{lang_cap}**. Some input context (like concept name, description, chapter list, previous summary, API call info) might already be in {lang_cap}, but you MUST translate ALL other generated content including explanations, examples, technical terms, and potentially code comments into {lang_cap}. DO NOT use English anywhere except in code syntax, required proper nouns, or when specified. The entire output MUST be in {lang_cap}.\n\n"
            concept_details_note = f" (Note: Provided in {lang_cap})"
            structure_note = f" (Note: Chapter names might be in {lang_cap})"
            prev_summary_note = f" (Note: This summary might be in {lang_cap})"
            instruction_lang_note = f" (in {lang_cap})"
            mermaid_lang_note = f" (Use {lang_cap} for labels/text if appropriate)"
            code_comment_note = f" (Translate to {lang_cap} if possible, otherwise keep minimal English for clarity)"
            link_lang_note = (
                f" (Use the {lang_cap} chapter title from the structure above)"
            )
            tone_note = f" (appropriate for {lang_cap} readers)"
            api_info_note = f" (Note: API call details might contain elements in {lang_cap})"

        # Prepare API call information for the prompt
        api_call_prompt_lines = []
        if api_calls_for_chapter:
            api_call_prompt_lines.append(f'''API Call Information for files related to "{abstraction_name}"{api_info_note}:''')
            for api_info in api_calls_for_chapter:
                api_call_prompt_lines.append(f"In file `{api_info['file_path']}`:")
                if api_info['api_calls']:
                    for call in api_info['api_calls']:
                        calling_func = call.get('calling_function_name', 'N/A')
                        endpoint = call.get('api_endpoint', 'N/A')
                        method = call.get('http_method', 'N/A')
                        req_params = call.get('request_parameters', 'N/A')
                        resp_usage = call.get('response_usage', 'N/A')
                        api_call_prompt_lines.append(f"- Function: `{calling_func}` calls API: `{endpoint}` (Method: {method})")
                        api_call_prompt_lines.append(f"  Request Params: {req_params}")
                        api_call_prompt_lines.append(f"  Response Usage: {resp_usage}")
                else:
                    api_call_prompt_lines.append("- No specific API calls found in the automated analysis for this file section, or the file was not a frontend file type.")
            api_call_prompt_lines.append("\n")
        api_call_prompt_section = "\n".join(api_call_prompt_lines)

        prompt_template = f"""
{language_instruction}Write a very beginner-friendly tutorial chapter (in Markdown format) for the project {{project_name}} about the concept: "{{abstraction_name}}". This is Chapter {{chapter_num}}.

Concept Details{{concept_details_note}}:
- Name: {{abstraction_name}}
- Description:
{{abstraction_description}}

Complete Tutorial Structure{{structure_note}}:
{{full_chapter_listing}}

Context from previous chapters{{prev_summary_note}}:
{{previous_chapters_summary}}

Relevant Code Snippets (Code itself remains unchanged):
{{file_context_str}}

{{api_call_prompt_section}}
Instructions for the chapter (Generate content in {language.capitalize()} unless specified otherwise):
- Start with a clear heading (e.g., `# Chapter {{chapter_num}}: {{abstraction_name}}`). Use the provided concept name.

- If this is not the first chapter, begin with a brief transition from the previous chapter{{instruction_lang_note}}, referencing it with a proper Markdown link using its name{{link_lang_note}}.

- Begin with a high-level motivation explaining what problem this abstraction solves{{instruction_lang_note}}. Start with a central use case as a concrete example. The whole chapter should guide the reader to understand how to solve this use case. Make it very minimal and friendly to beginners.

- If the abstraction is complex, break it down into key concepts. Explain each concept one-by-one in a very beginner-friendly way{{instruction_lang_note}}.

- Explain how to use this abstraction to solve the use case{{instruction_lang_note}}. Give example inputs and outputs for code snippets (if the output isn't values, describe at a high level what will happen{{instruction_lang_note}}).

- Each code block should be BELOW 10 lines! If longer code blocks are needed, break them down into smaller pieces and walk through them one-by-one. Aggresively simplify the code to make it minimal. Use comments{{code_comment_note}} to skip non-important implementation details. Each code block should have a beginner friendly explanation right after it{{instruction_lang_note}}.

- Describe the internal implementation to help understand what's under the hood{{instruction_lang_note}}. First provide a non-code or code-light walkthrough on what happens step-by-step when the abstraction is called{{instruction_lang_note}}. It's recommended to use a simple sequenceDiagram with a dummy example - keep it minimal with at most 5 participants to ensure clarity. If participant name has space, use: `participant QP as Query Processing`. {{mermaid_lang_note}}.

- Then dive deeper into code for the internal implementation with references to files. Provide example code blocks, but make them similarly simple and beginner-friendly. Explain{{instruction_lang_note}}.

- **If API call information is provided above, integrate it naturally into the explanations.** For example, when discussing a function or a piece of code that makes an API call, describe the endpoint, parameters, and how the response is used, based on the provided API call details. This should be part of the regular explanation, not a separate, disconnected section. {{instruction_lang_note}}

- IMPORTANT: When you need to refer to other core abstractions covered in other chapters, ALWAYS use proper Markdown links like this: [Chapter Title](filename.md). Use the Complete Tutorial Structure above to find the correct filename and the chapter title{{link_lang_note}}. Translate the surrounding text.

- Use mermaid diagrams to illustrate complex concepts (```mermaid``` format). {{mermaid_lang_note}}.

- Heavily use analogies and examples throughout{{instruction_lang_note}} to help beginners understand.

- End the chapter with a brief conclusion that summarizes what was learned{{instruction_lang_note}} and provides a transition to the next chapter{{instruction_lang_note}}. If there is a next chapter, use a proper Markdown link: [Next Chapter Title](next_chapter_filename){{link_lang_note}}.

- Ensure the tone is welcoming and easy for a newcomer to understand{{tone_note}}.

- Output *only* the Markdown content for this chapter.

Now, directly provide a super beginner-friendly Markdown output (DON'T need ```markdown``` tags):
"""
        prompt = prompt_template.format(
            project_name=project_name,
            abstraction_name=abstraction_name,
            chapter_num=chapter_num,
            concept_details_note=concept_details_note,
            abstraction_description=abstraction_description,
            structure_note=structure_note,
            full_chapter_listing=item["full_chapter_listing"],
            prev_summary_note=prev_summary_note,
            previous_chapters_summary=previous_chapters_summary if previous_chapters_summary else "This is the first chapter.",
            file_context_str=file_context_str if file_context_str else "No specific code snippets provided for this abstraction.",
            api_call_prompt_section=api_call_prompt_section,
            instruction_lang_note=instruction_lang_note,
            link_lang_note=link_lang_note,
            code_comment_note=code_comment_note,
            mermaid_lang_note=mermaid_lang_note,
            tone_note=tone_note
        )

        chapter_content = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0)) # Use cache only if enabled and not retrying
        # Basic validation/cleanup
        actual_heading = f"# Chapter {chapter_num}: {abstraction_name}"  # Use potentially translated name
        if not chapter_content.strip().startswith(f"# Chapter {chapter_num}"):
            # Add heading if missing or incorrect, trying to preserve content
            lines = chapter_content.strip().split("\n")
            if lines and lines[0].strip().startswith(
                "#"
            ):  # If there's some heading, replace it
                lines[0] = actual_heading
                chapter_content = "\n".join(lines)
            else:  # Otherwise, prepend it
                chapter_content = f"{actual_heading}\n\n{chapter_content}"

        # Add the generated content to our temporary list for the next iteration's context
        self.chapters_written_so_far.append(chapter_content)

        return chapter_content  # Return the Markdown string (potentially translated)

    def post(self, shared, prep_res, exec_res_list):
        # exec_res_list contains the generated Markdown for each chapter, in order
        shared["chapters"] = exec_res_list
        # Clean up the temporary instance variable
        del self.chapters_written_so_far
        print(f"Finished writing {len(exec_res_list)} chapters.")


class CombineTutorial(Node):
    def prep(self, shared):
        project_name = shared["project_name"]
        output_base_dir = shared.get("output_dir", "output")  # Default output dir
        output_path = os.path.join(output_base_dir, project_name)
        repo_url = shared.get("repo_url")  # Get the repository URL
        api_documentation_md = shared.get("api_documentation_md", "") # Get API documentation

        # Get potentially translated data
        relationships_data = shared[
            "relationships"
        ]  # {"summary": str, "details": [...]} -> summary/label potentially translated
        chapter_order = shared["chapter_order"]  # indices
        abstractions = shared[
            "abstractions"
        ]  # list of dicts -> name/description potentially translated
        chapters_content = shared[
            "chapters"
        ]  # list of strings -> content potentially translated

        # --- Generate Mermaid Diagram ---
        mermaid_lines = ["flowchart TD"]
        for i, abstr in enumerate(abstractions):
            node_id = f"A{i}"
            sanitized_name = abstr["name"].replace('"', "")
            node_label = sanitized_name
            mermaid_lines.append(
                f'    {node_id}["{node_label}"]'
            )
        for rel in relationships_data["details"]:
            from_node_id = f"A{rel['from']}"
            to_node_id = f"A{rel['to']}"
            edge_label = (
                rel["label"].replace('"', "").replace("\n", " ")
            )
            max_label_len = 30
            if len(edge_label) > max_label_len:
                edge_label = edge_label[: max_label_len - 3] + "..."
            mermaid_lines.append(
                f'    {from_node_id} -- "{edge_label}" --> {to_node_id}'
            )
        mermaid_diagram = "\n".join(mermaid_lines)

        # --- Prepare index.md content ---
        index_content = f"# Tutorial: {project_name}\n\n"
        index_content += f"{relationships_data['summary']}\n\n"
        index_content += f"**Source Repository:** [{repo_url}]({repo_url})\n\n"

        # Add link to API Documentation if it exists
        api_doc_filename = "api_reference.md"
        if api_documentation_md and api_documentation_md.strip():
            index_content += f"## API Reference\n\n"
            index_content += f"See the [API Reference]({api_doc_filename}) for details on backend endpoints.\n\n"

        index_content += "## Codebase Abstractions Diagram\n\n"
        index_content += "```mermaid\n"
        index_content += mermaid_diagram + "\n"
        index_content += "```\n\n"
        index_content += f"## Chapters\n\n"

        chapter_files = []
        for i, abstraction_index in enumerate(chapter_order):
            if 0 <= abstraction_index < len(abstractions) and i < len(chapters_content):
                abstraction_name = abstractions[abstraction_index]["name"]
                safe_name = "".join(
                    c if c.isalnum() else "_" for c in abstraction_name
                ).lower()
                filename = f"{i+1:02d}_{safe_name}.md"
                index_content += f"{i+1}. [{abstraction_name}]({filename})\n"
                chapter_content = chapters_content[i]
                if not chapter_content.endswith("\n\n"):
                    chapter_content += "\n\n"
                chapter_content += f"---\n\nGenerated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)"
                chapter_files.append({"filename": filename, "content": chapter_content})
            else:
                print(
                    f"Warning: Mismatch between chapter order, abstractions, or content at index {i} (abstraction index {abstraction_index}). Skipping file generation for this entry."
                )
        index_content += f"\n\n---\n\nGenerated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)"

        return {
            "output_path": output_path,
            "index_content": index_content,
            "chapter_files": chapter_files,
            "api_documentation_md": api_documentation_md, # Pass content for writing
            "api_doc_filename": api_doc_filename if api_documentation_md and api_documentation_md.strip() else None
        }

    def exec(self, prep_res):
        output_path = prep_res["output_path"]
        index_content = prep_res["index_content"]
        chapter_files = prep_res["chapter_files"]
        api_documentation_md = prep_res["api_documentation_md"]
        api_doc_filename = prep_res["api_doc_filename"]

        print(f"Combining tutorial into directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)

        index_filepath = os.path.join(output_path, "index.md")
        with open(index_filepath, "w", encoding="utf-8") as f:
            f.write(index_content)
        print(f"  - Wrote {index_filepath}")

        # Write API documentation file if content exists
        if api_doc_filename and api_documentation_md:
            api_doc_filepath = os.path.join(output_path, api_doc_filename)
            with open(api_doc_filepath, "w", encoding="utf-8") as f:
                f.write(api_documentation_md)
            print(f"  - Wrote {api_doc_filepath}")

        for chapter_info in chapter_files:
            chapter_filepath = os.path.join(output_path, chapter_info["filename"])
            with open(chapter_filepath, "w", encoding="utf-8") as f:
                f.write(chapter_info["content"])
            print(f"  - Wrote {chapter_filepath}")

        return output_path

    def post(self, shared, prep_res, exec_res):
        shared["final_output_dir"] = exec_res  # Store the output path
        print(f"\nTutorial generation complete! Files are in: {exec_res}")

