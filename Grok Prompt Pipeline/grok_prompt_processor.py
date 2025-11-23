"""
Grok Prompt Processor with Real-time Validation
Processes mentorship data and queries Grok API to identify events, green flags, and red flags.
"""

import os
import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from nltk import sent_tokenize
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration
XAI_API_KEY = os.getenv("XAI_API_KEY", "xai-5q7CtzPajDnu5Ucaic4kCbaMXcskpnYt2In72q49rGBrM49T10gZ0kCkTsAVw0hqrnoY2kbszKNg7IlK")
os.environ["XAI_API_KEY"] = XAI_API_KEY

# Expected event columns for validation
EXPECTED_EVENT_COLUMNS = [
    'Match closure Discussed',
    'Changing Match Type',
    'COVID impact',
    'Child/Family: Feels incompatible with volunteer',
    'Child/Family: Moved',
    'Child/Family: Lost contact with agency',
    'Child/Family: Lost contact with volunteer/agency',
    'Child/Family: Lost contact with volunteer',
    'Child/Family: Moved out of service',
    'Child/Family: Unrealistic expectations',
    'Child/Family: Time constraints',
    'Child/Family: Infraction of match rules/agency policies',
    'Child/Family: Moved within service area',
    'Child: Graduated',
    'Child: Transportation Issues',
    'Child: Changed school/site',
    'Child: Lost interest',
    'Child: Family structure changed',
    'Child: Severity of challenges',
    'Volunteer: Transportation Issues',
    'Volunteer: Moved out of service area',
    'Volunteer: Moved within service area',
    'Volunteer: Lost contact with agency',
    'Volunteer: Lost contact with child/agency',
    'Volunteer: Feels incompatible with child/family',
    'Volunteer: Time constraint',
    'Volunteer: Deceased',
    'Volunteer: Lost contact with child/family',
    'Volunteer: Infraction of match rules/agency policies',
    'Volunteer: Unrealistic expectations',
    'Volunteer: Pregnancy',
    'Volunteer: Changed workplace/school partnership',
    'Agency: Challenges with program/partnership',
    'Agency: Concern with Volunteer re: child safety'
]

# Prompt templates
CONTEXT = '''
Follow prompt instruction explicitly without exceptions. You are a machine processing text. Your only task is to identify potential events, green flags (e.g., factors likely to enhance relationship quality and duration) and red flags (e.g., risks of early termination or poor outcomes) in the mentorship program by big brothers big sisters of america.

Background:
BBB or Agency = Big brothers' big sisters of America organization
LB/LS = Little Brother/Sister (Mentee Or Child)
BB/BS = Big Brother/Sister (Mentor Or Volunteer)
MEC or MSC = Match coordinator from BBB
PG = Parent of LB/LS
Current Service area = Minnesota

Flags :
Green Flag indicates any events falling near to the categories below , with a positive impact on match
Red Flag indicates any events falling near to the categories below , with a negative impact on match

Green Flags to Detect:
•Any positive Events identified in the Rationale for Match
•Indication that Mentor completed BBB training pre-match
•Commitment to 18-month match
•Shared interests/preferences in match
•Monthly in-person/phone support from agency to mentor, youth, parent
•High mentor satisfaction, realistic expectations
•Youth reports positive relationship, frequent meetings
•Demographic alignment (race, gender, religion)
•Close geographic proximity or good transportation access
•Positive youth traits (5 Cs: competence, confidence, connection, care, character)
•Older, experienced mentor with empathy, flexibility, multicultural competence
•Younger mentee (elementary–early adolescence), good relational history

Red Flags to Detect:
•No pre-match training or ongoing support
•Mismatched interests, ignored mentor preferences
•Infrequent/superficial staff check-ins (<6 min)
•Mentor frustration, unrealistic expectations, youth resistance
•No closure plan for early termination
•Match ends <6 months (34–50% risk)
•Younger mentor (18–25), negative attitudes, low commitment
•Older mentee seeking autonomy, severe risk factors
•No monthly staff support (email-only contact)
•Inadequate BBB training, excessive/scanty staff involvement
•Parental dissatisfaction/interference
•Match ends <13–18 months

'''

OUTPUT_GUIDANCE = '''
Response Guidance
Events to Detect (Used Exclusively for JSON "events" Field):
(Ensure detected events are exactly as listed below)

Match-Level Events:
Match closure Discussed,
Changing Match Type,
COVID impact

Child/Family-Related Events:
Child/Family: Feels incompatible with volunteer,
Child/Family: Moved,
Child/Family: Lost contact with agency,
Child/Family: Lost contact with volunteer/agency,
Child/Family: Lost contact with volunteer,
Child/Family: Moved out of service,
Child/Family: Unrealistic expectations,
Child/Family: Time constraints,
Child/Family: Infraction of match rules/agency policies
Child/Family: Moved within service area
Child: Graduated
Child: Transportation Issues
Child: Changed school/site
Child: Lost interest
Child: Family structure changed
Child: Severity of challenges

Volunteer-Related Events:
Volunteer: Transportation Issues,
Volunteer: Moved out of service area,
Volunteer: Moved within service area,
Volunteer: Lost contact with agency,
Volunteer: Lost contact with child/agency,
Volunteer: Feels incompatible with child/family,
Volunteer: Time constraints,
Volunteer: Deceased,
Volunteer: Lost contact with child/family,
Volunteer: Infraction of match rules/agency policies,
Volunteer: Unrealistic expectations,
Volunteer: Pregnancy,
Volunteer: Changed workplace/school partnership,

Agency-Level Events:
Agency: Challenges with program/partnership
Agency: Concern with Volunteer re: child safety

JSON Response Format (Strictly Follow This Structure)

Always return a valid JSON object containing:
green_flag_count: Number of detected positive indicators (if any)
red_flag_count: Number of detected concerning events
events: A dictionary where:
Keys are detected event names (must match exactly from the provided list)
Values are assigned severity scores (1–5)

Example JSON Response:
[
  {
    "green_flag_count": 2,
    "red_flag_count": 3,
    "events": {
      "Child/Family: Unrealistic expectations": 3,
      "Volunteer: Unrealistic expectations": 4,
      "Agency: Concern with Volunteer re: child safety": 5
    }
  }
]
'''

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_file_exists(filepath: str) -> bool:
    """Validate that a file exists."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return True

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame contains required columns."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True

def validate_json_structure(data: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Validate JSON response structure matches expected format."""
    if not isinstance(data, list):
        return False, "Response must be a list"
    
    if len(data) == 0:
        return False, "Response list is empty"
    
    for item in data:
        if not isinstance(item, dict):
            return False, "Each item must be a dictionary"
        
        # Check required fields
        required_fields = ['green_flag_count', 'red_flag_count', 'events']
        for field in required_fields:
            if field not in item:
                return False, f"Missing required field: {field}"
        
        # Validate field types
        if not isinstance(item['green_flag_count'], (int, float)):
            return False, "green_flag_count must be numeric"
        if not isinstance(item['red_flag_count'], (int, float)):
            return False, "red_flag_count must be numeric"
        if not isinstance(item['events'], dict):
            return False, "events must be a dictionary"
        
        # Validate event names and severity scores
        for event_name, severity in item['events'].items():
            if event_name not in EXPECTED_EVENT_COLUMNS:
                return False, f"Unknown event name: {event_name}"
            if not isinstance(severity, (int, float)) or not (1 <= severity <= 5):
                return False, f"Severity score for {event_name} must be between 1 and 5, got {severity}"
    
    return True, None

def validate_api_key() -> bool:
    """Validate that API key is set."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable is not set.")
    if not api_key.startswith("xai-"):
        raise ValueError("XAI_API_KEY appears to be invalid (should start with 'xai-')")
    return True

def validate_text_input(text: str, field_name: str) -> bool:
    """Validate text input is not empty."""
    if not isinstance(text, str):
        raise TypeError(f"{field_name} must be a string")
    if len(text.strip()) == 0:
        raise ValueError(f"{field_name} cannot be empty")
    return True

# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================

def shorten_text(text: str) -> str:
    """
    Tokenizer to shorten sentences as well as remove unwanted stuff like URLs and spaces.
    Validates input and output.
    """
    # Validate input
    validate_text_input(text, "text")
    
    # Remove URLs
    text = re.sub(r'http[s]?://[^\s]+', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove sequences of underscores longer than 3
    text = re.sub(r'_{4,}', '', text)
    
    # Split into dated blocks
    blocks = text.split('\n\n')
    shortened_blocks = []
    
    for block in blocks:
        # Extract the date (assuming YYYY-MM-DD format)
        match = re.match(r'(\d{4}-\d{2}-\d{2})', block)
        if not match:
            continue
        date = match.group(1)
        content = block[len(date):].strip()
        
        # Split content into lines
        lines = content.split('\n')
        filtered_lines = []
        
        for line in lines:
            if "Answer:" in line:
                question, answer = line.split("Answer:", 1)
                answer = answer.strip()
                if answer.lower() not in ['na', 'n/a', '-', '_', '__', '___', '']:
                    filtered_lines.append(f"{question.strip()} Answer: {answer}")
            else:
                filtered_lines.append(line.strip())
        
        if not filtered_lines:
            shortened_content = content.split('\n')[0]
        else:
            combined_content = ' '.join(filtered_lines)
            sentences = sent_tokenize(combined_content)
            key_sentences = [s for s in sentences if "See notes" not in s][:2]
            shortened_content = ' '.join(key_sentences) if key_sentences else combined_content.split('\n')[0]
        
        shortened_blocks.append(f"{date} {shortened_content}")
    
    result = '\n\n'.join(shortened_blocks) if shortened_blocks else text.split('\n\n')[0]
    
    # Validate output
    if len(result.strip()) == 0:
        print(f"Warning: shorten_text produced empty result for input: {text[:50]}...")
        return text[:100]  # Return first 100 chars as fallback
    
    return result

# ============================================================================
# API FUNCTIONS
# ============================================================================

def parse_json_message(json_string: str) -> List[Dict[str, Any]]:
    """
    Parse JSON message from API response with validation.
    Handles malformed JSON by attempting to fix it.
    """
    if not isinstance(json_string, str):
        raise TypeError("json_string must be a string")
    
    if len(json_string.strip()) == 0:
        raise ValueError("json_string cannot be empty")
    
    try:
        brace_position = json_string.index('[')
        json_string = json_string[brace_position:]
    except ValueError:
        raise ValueError("Input does not start with '[': invalid JSON array format.")
    
    original_string = json_string
    attempts = 0
    max_attempts = len(json_string)
    
    while attempts < max_attempts:
        if not json_string:
            raise ValueError("Couldn't fix JSON: string became empty")
        try:
            data = json.loads(json_string + "]")
            # Validate the parsed structure
            is_valid, error_msg = validate_json_structure(data)
            if not is_valid:
                raise ValueError(f"Invalid JSON structure: {error_msg}")
            return data
        except json.JSONDecodeError:
            json_string = json_string[:-1]
            attempts += 1
            continue
        except ValueError as e:
            # Re-raise validation errors
            raise
    
    raise ValueError(f"Couldn't fix JSON after {max_attempts} attempts. Original: {original_string[:200]}")

# Global state for conversation management
count = 0
conversation_history: List[Dict[str, str]] = []
last_matchid: Optional[str] = None

def query_grok(query_str: str, matchid: str, rationale: str) -> str:
    """
    Query Grok API with conversation history management.
    Includes real-time validation of inputs and outputs.
    """
    global count, conversation_history, last_matchid
    
    # Validate inputs
    validate_text_input(query_str, "query_str")
    validate_text_input(matchid, "matchid")
    validate_text_input(rationale, "rationale")
    validate_api_key()
    
    # Retrieve the API key
    api_key = os.getenv("XAI_API_KEY")
    
    # Initialize the OpenAI client
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
    
    # Check if matchid has changed
    if matchid != last_matchid:
        print("---------------------------- New Convo Context --------------------------")
        conversation_history = [{"role": "system", "content": CONTEXT + rationale}]
        last_matchid = matchid
        print(f"Match ID: {matchid}")
        print(f"Context length: {len(conversation_history[0]['content'])} chars")
    
    # Append the query to the conversation history
    user_message = {"role": "user", "content": query_str + OUTPUT_GUIDANCE}
    conversation_history.append(user_message)
    
    # Validate conversation history
    if len(conversation_history) > 100:
        print("Warning: Conversation history exceeds 100 messages, may cause issues")
    
    # Make API call with the full conversation history
    try:
        response = client.chat.completions.create(
            model="grok-beta",
            messages=conversation_history,
            max_tokens=1000
        )
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}")
    
    # Validate response
    if not response or not response.choices:
        raise ValueError("Empty response from API")
    
    # Extract and append the assistant's response
    count += 1
    answer = response.choices[0].message.content
    
    if not answer:
        raise ValueError("Empty answer from API response")
    
    conversation_history.append({"role": "assistant", "content": answer})
    
    print(f"Processed: {count}")
    print("------------------------------")
    print(f"Response preview: {answer[:200]}...")
    
    token_count = response.usage.total_tokens if response.usage else 0
    print("------------------------------")
    print(f"Token Count: {token_count}")
    
    # Validate token usage
    if token_count > 900:
        print(f"Warning: Token usage ({token_count}) is close to max_tokens (1000)")
    
    return answer

def process_row(row: pd.Series, error_df: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
    """
    Process a single row with validation and error handling.
    Returns parsed JSON response or None if error occurred.
    """
    try:
        # Validate required columns exist
        required_cols = ["Rationale for Match", "Dated Short Notes", "Match ID 18Char"]
        for col in required_cols:
            if col not in row.index:
                raise ValueError(f"Missing column: {col}")
        
        rationale = str(row["Rationale for Match"])
        query = str(row["Dated Short Notes"])
        matchid = str(row["Match ID 18Char"])
        
        # Validate inputs
        validate_text_input(rationale, "Rationale for Match")
        validate_text_input(query, "Dated Short Notes")
        validate_text_input(matchid, "Match ID 18Char")
        
        # Query Grok API
        json_response = query_grok(query, matchid, rationale)
        
        # Parse and validate JSON response
        parsed_response = parse_json_message(json_response)
        
        # Additional validation: check event names match expected list
        for item in parsed_response:
            if 'events' in item:
                for event_name in item['events'].keys():
                    if event_name not in EXPECTED_EVENT_COLUMNS:
                        print(f"Warning: Unexpected event name detected: {event_name}")
        
        print("------------------------------")
        print(f"Parsed response: {parsed_response}")
        
        return parsed_response
        
    except (ValueError, json.JSONDecodeError, RuntimeError, TypeError) as e:
        error_msg = str(e)
        print(f"Error processing row: {error_msg}")
        
        # Add to error DataFrame
        matchid = str(row.get("Match ID 18Char", "UNKNOWN"))
        rationale = str(row.get("Rationale for Match", ""))
        query = str(row.get("Dated Short Notes", ""))
        
        error_row = pd.DataFrame({
            "Match ID 18Char": [matchid],
            "Rationale for Match": [rationale],
            "Dated Short Notes": [query],
            "Error": [error_msg]
        })
        
        error_df = pd.concat([error_df, error_row], ignore_index=True)
        return None

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def load_and_prepare_data(input_file: str) -> pd.DataFrame:
    """
    Load and prepare data with validation.
    """
    print(f"Loading data from: {input_file}")
    validate_file_exists(input_file)
    
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file: {str(e)}")
    
    # Validate required columns
    required_columns = ['Match ID 18Char', 'Completion Date', 'Match Support Contact Notes']
    validate_dataframe_columns(df, required_columns)
    
    print(f"Loaded {len(df)} rows")
    
    # Sort the data
    df_sorted = df.sort_values(by=['Match ID 18Char', 'Completion Date'], ascending=[True, True])
    df_sorted = df_sorted.groupby('Match ID 18Char').tail(10)
    
    print(f"After filtering: {len(df_sorted)} rows")
    
    # Fill Empty Data
    df_sorted['Match Support Contact Notes'] = df_sorted['Match Support Contact Notes'].fillna('No Updates').astype(str)
    
    # Ensure 'Completion Date' is in datetime format
    df_sorted['Completion Date'] = pd.to_datetime(df_sorted['Completion Date'], errors='coerce')
    
    # Validate date conversion
    null_dates = df_sorted['Completion Date'].isna().sum()
    if null_dates > 0:
        print(f"Warning: {null_dates} rows have invalid dates")
    
    # Process text
    print("Processing text...")
    df_sorted['Short Notes'] = df_sorted['Match Support Contact Notes'].apply(shorten_text)
    df_sorted['Dated Short Notes'] = df_sorted['Completion Date'].astype(str) + ' ' + df_sorted['Short Notes'].fillna('').astype(str)
    
    # Validate processed text
    empty_notes = (df_sorted['Dated Short Notes'].str.strip() == '').sum()
    if empty_notes > 0:
        print(f"Warning: {empty_notes} rows have empty processed notes")
    
    print("Data preparation complete")
    return df_sorted

def process_dataframe(df_sorted: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process DataFrame with API calls and return results with error tracking.
    """
    # Initialize error_df
    error_df = pd.DataFrame(columns=["Match ID 18Char", "Rationale for Match", "Dated Short Notes", "Error"])
    
    # Validate required columns
    required_cols = ["Rationale for Match", "Dated Short Notes", "Match ID 18Char"]
    validate_dataframe_columns(df_sorted, required_cols)
    
    print(f"Processing {len(df_sorted)} rows...")
    
    # Process each row
    df_sorted["JSON Response"] = df_sorted.apply(
        lambda row: process_row(row, error_df), axis=1
    )
    
    # Validate results
    successful = df_sorted["JSON Response"].notna().sum()
    failed = df_sorted["JSON Response"].isna().sum()
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if len(error_df) > 0:
        print(f"\nErrors encountered: {len(error_df)}")
        print(error_df.head())
    
    return df_sorted, error_df

def flatten_json_responses(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten JSON responses into columns with validation.
    """
    print("Flattening JSON responses...")
    
    # Validate JSON Response column exists
    if "JSON Response" not in df_sorted.columns:
        raise ValueError("JSON Response column not found in DataFrame")
    
    # Normalize JSON
    flat_df = pd.json_normalize(df_sorted["JSON Response"])
    
    if len(flat_df) == 0:
        raise ValueError("No data to flatten")
    
    # Validate structure
    if 'events' not in flat_df.columns:
        raise ValueError("'events' column not found in normalized data")
    
    # Extract events
    events_df = flat_df["events"].apply(pd.Series).fillna(0)
    
    # Add missing columns
    for col in EXPECTED_EVENT_COLUMNS:
        if col not in events_df.columns:
            events_df[col] = 0
    
    # Validate all expected columns are present
    missing_cols = [col for col in EXPECTED_EVENT_COLUMNS if col not in events_df.columns]
    if missing_cols:
        raise ValueError(f"Missing event columns after processing: {missing_cols}")
    
    # Drop events column and join
    flat_df = flat_df.drop(columns=["events"]).join(events_df)
    
    # Convert to int
    flat_df[EXPECTED_EVENT_COLUMNS] = flat_df[EXPECTED_EVENT_COLUMNS].astype(int)
    
    # Validate data types
    for col in EXPECTED_EVENT_COLUMNS:
        if not pd.api.types.is_integer_dtype(flat_df[col]):
            print(f"Warning: Column {col} is not integer type, converting...")
            flat_df[col] = pd.to_numeric(flat_df[col], errors='coerce').fillna(0).astype(int)
    
    print(f"Flattened {len(flat_df)} rows with {len(EXPECTED_EVENT_COLUMNS)} event columns")
    
    return flat_df

def create_final_dataframe(df_sorted: pd.DataFrame, flat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final DataFrame by merging and cleaning.
    """
    print("Creating final DataFrame...")
    
    # Merge
    df_final = pd.concat([df_sorted, flat_df], axis=1)
    
    # Drop intermediate columns
    columns_to_drop = ["JSON Response", "Short Notes", "Dated Short Notes"]
    for col in columns_to_drop:
        if col in df_final.columns:
            df_final = df_final.drop(columns=[col])
    
    # Validate final structure
    print(f"Final DataFrame shape: {df_final.shape}")
    print(f"Final DataFrame columns: {len(df_final.columns)}")
    
    return df_final

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(input_file: str = "MUDAC/External Data/Test-Truncated-Restated.xlsx",
         output_file: str = "Grok Prompt Pipeline/output_df.xlsx",
         intermediate_file: str = "Grok Prompt Pipeline/combined_notes.xlsx",
         error_file: str = "Grok Prompt Pipeline/errors.xlsx"):
    """
    Main execution function with full pipeline and validation.
    """
    try:
        print("=" * 80)
        print("Grok Prompt Processor - Starting")
        print("=" * 80)
        
        # Step 1: Load and prepare data
        print("\n[Step 1] Loading and preparing data...")
        df_sorted = load_and_prepare_data(input_file)
        
        # Save intermediate result
        print(f"\nSaving intermediate result to: {intermediate_file}")
        os.makedirs(os.path.dirname(intermediate_file), exist_ok=True)
        df_sorted.to_excel(intermediate_file, index=False)
        print("Intermediate file saved")
        
        # Step 2: Process with API
        print("\n[Step 2] Processing with Grok API...")
        df_sorted, error_df = process_dataframe(df_sorted)
        
        # Save error log
        if len(error_df) > 0:
            print(f"\nSaving error log to: {error_file}")
            error_df.to_excel(error_file, index=False)
            print("Error log saved")
        
        # Step 3: Flatten JSON responses
        print("\n[Step 3] Flattening JSON responses...")
        flat_df = flatten_json_responses(df_sorted)
        
        # Step 4: Create final DataFrame
        print("\n[Step 4] Creating final DataFrame...")
        df_final = create_final_dataframe(df_sorted, flat_df)
        
        # Step 5: Save output
        print(f"\n[Step 5] Saving output to: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_final.to_excel(output_file, index=False)
        print("Output file saved")
        
        # Final summary
        print("\n" + "=" * 80)
        print("Processing Complete!")
        print("=" * 80)
        print(f"Total rows processed: {len(df_final)}")
        print(f"Output file: {output_file}")
        if len(error_df) > 0:
            print(f"Errors encountered: {len(error_df)} (see {error_file})")
        print("=" * 80)
        
        return df_final, error_df
        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # You can customize these paths
    main(
        input_file="MUDAC/External Data/Test-Truncated-Restated.xlsx",
        output_file="Grok Prompt Pipeline/output_df.xlsx",
        intermediate_file="Grok Prompt Pipeline/combined_notes.xlsx",
        error_file="Grok Prompt Pipeline/errors.xlsx"
    )

