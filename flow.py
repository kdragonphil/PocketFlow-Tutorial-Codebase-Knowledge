from pocketflow import Flow
# Import all node classes from nodes.py
from nodes import (
    FetchRepo,
    IdentifyAbstractions,
    AnalyzeRelationships,
    OrderChapters,
    WriteChapters,
    CombineTutorial
)

def create_tutorial_flow():
    """Creates and returns the codebase tutorial generation flow."""

    # Instantiate nodes
    fetch_repo = FetchRepo()
    analyze_fastapi_endpoints = AnalyzeFastAPIEndpoints(max_retries=3, wait=15)
    generate_api_documentation = GenerateAPIDocumentation(max_retries=3, wait=15)
    identify_abstractions = IdentifyAbstractions(max_retries=5, wait=20)
    analyze_api_calls = AnalyzeAPICalls(max_retries=3, wait=15)
    analyze_relationships = AnalyzeRelationships(max_retries=5, wait=20)
    order_chapters = OrderChapters(max_retries=5, wait=20)
    write_chapters = WriteChapters(max_retries=5, wait=20) # This is a BatchNode
    combine_tutorial = CombineTutorial()

    # Connect nodes in sequence based on the design
    fetch_repo >> analyze_fastapi_endpoints
    analyze_fastapi_endpoints >> generate_api_documentation
    generate_api_documentation >> identify_abstractions
    identify_abstractions >> analyze_api_calls
    analyze_api_calls >> analyze_relationships
    analyze_relationships >> order_chapters
    order_chapters >> write_chapters
    write_chapters >> combine_tutorial

    # Create the flow starting with FetchRepo
    tutorial_flow = Flow(start=fetch_repo)

    return tutorial_flow
