"""Helper functions for LLM"""

import json
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from utils.progress import progress

T = TypeVar('T', bound=BaseModel)

def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory = None
) -> T:
    """
    Makes an LLM call with retry logic, handling both Deepseek and non-Deepseek models.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure
        
    Returns:
        An instance of the specified Pydantic model
    """
    from llm.models import get_model, get_model_info, ModelProvider
    
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)
    
    # For models that support structured output, use it
    needs_json_extraction = (model_info and model_info.is_deepseek()) or model_provider == ModelProvider.OLLAMA
    
    if not needs_json_extraction:
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )
    
    # Special handling for specific agents to ensure we get valid output
    is_portfolio_manager = pydantic_model.__name__ == "PortfolioManagerOutput"
    is_bill_ackman = pydantic_model.__name__ == "BillAckmanSignal"
    is_warren_buffett = pydantic_model.__name__ == "WarrenBuffettSignal"
    is_charlie_munger = pydantic_model.__name__ == "CharlieMungerSignal"
    
    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)
            
            # For models that need JSON extraction, we need to extract and parse the JSON manually
            if needs_json_extraction:
                if is_portfolio_manager:
                    # Special handling for portfolio manager
                    parsed_result = extract_portfolio_decisions(result.content)
                elif is_warren_buffett or is_charlie_munger or is_bill_ackman:
                    # Special handling for Warren Buffett, Charlie Munger, and Bill Ackman signals
                    parsed_result = extract_investment_signal(result.content)
                else:
                    parsed_result = extract_json_from_response(result.content)
                
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result
                
        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")
            
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)

def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    # Get field names and types from the model
    fields = model_class.__annotations__
    
    # Create a dictionary with default values for each field
    default_values = {}
    for field_name, field_type in fields.items():
        # Handle different field types with appropriate defaults
        if field_name == "signal" and hasattr(model_class, "__annotations__") and "signal" in model_class.__annotations__:
            # Special handling for signal field which is often a Literal type
            default_values[field_name] = "neutral"  # Safe default for signal fields
        elif field_name == "decisions" and field_type == dict:
            # Special handling for decisions field in PortfolioManagerOutput
            default_values[field_name] = {}
        elif field_type == str:
            default_values[field_name] = "No response available"
        elif field_type == int or field_type == float:
            default_values[field_name] = 0
        elif field_type == bool:
            default_values[field_name] = False
        elif field_type == list:
            default_values[field_name] = []
        elif field_type == dict:
            default_values[field_name] = {}
        else:
            # For complex types, use None
            default_values[field_name] = None
    
    # Create and return the model instance
    try:
        return model_class(**default_values)
    except Exception as e:
        print(f"Error creating default response: {e}")
        # If validation fails, try to create a minimal valid instance
        if hasattr(model_class, "__fields__"):
            minimal_values = {}
            for field_name, field in model_class.__fields__.items():
                if hasattr(field, "is_required") and field.is_required:
                    if field_name == "signal":
                        minimal_values[field_name] = "neutral"
                    elif field_name == "decisions":
                        minimal_values[field_name] = {}
                    elif field.type_ == str:
                        minimal_values[field_name] = "No response available"
                    elif field.type_ == int or field.type_ == float:
                        minimal_values[field_name] = 0
                    elif field.type_ == bool:
                        minimal_values[field_name] = False
                    elif field.type_ == list:
                        minimal_values[field_name] = []
                    elif field.type_ == dict:
                        minimal_values[field_name] = {}
            return model_class(**minimal_values)
        
        # Last resort fallback
        raise e

def extract_json_from_response(content: str) -> dict:
    """Extracts JSON from model response, handling various formats."""
    # First, try to find JSON in markdown code blocks
    import re
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(json_block_pattern, content)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON in code blocks, try to find JSON-like structure in the text
    try:
        # Look for content between curly braces
        brace_pattern = r"\{[\s\S]*\}"
        brace_matches = re.findall(brace_pattern, content)
        if brace_matches:
            for match in brace_matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    
    # Try to extract key-value pairs from text for simple models
    try:
        # Look for patterns like "signal: bullish" or "confidence: 0.8"
        signal_match = re.search(r"signal:?\s*([a-zA-Z\s]+)", content, re.IGNORECASE)
        confidence_match = re.search(r"confidence:?\s*([\d\.]+)", content, re.IGNORECASE)
        reasoning_match = re.search(r"reasoning:?\s*(.+?)(?=\n\n|\n[A-Z]|$)", content, re.IGNORECASE | re.DOTALL)
        
        result = {}
        
        if signal_match:
            signal_value = signal_match.group(1).lower().strip()
            # Map common non-standard values to standard ones
            signal_mapping = {
                "cautious": "neutral",
                "hold": "neutral",
                "cautious bearish": "bearish",
                "cautious bullish": "bullish",
                "slightly bearish": "bearish",
                "slightly bullish": "bullish",
                "strongly bearish": "bearish",
                "strongly bullish": "bullish",
                "very bearish": "bearish",
                "very bullish": "bullish",
                "buy": "bullish",
                "sell": "bearish"
            }
            result["signal"] = signal_mapping.get(signal_value, signal_value)
            # Ensure signal is one of the allowed values
            if result["signal"] not in ["bullish", "bearish", "neutral"]:
                result["signal"] = "neutral"  # Default to neutral if invalid
                
        if confidence_match:
            try:
                result["confidence"] = float(confidence_match.group(1))
            except ValueError:
                result["confidence"] = 0.5  # Default confidence
        else:
            # If no confidence found, add a default
            result["confidence"] = 0.5
        
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        else:
            # If no reasoning found, add a default
            result["reasoning"] = "No detailed reasoning provided by the model."
        
        # Ensure all required fields are present
        if "signal" in result and "confidence" in result and "reasoning" in result:
            return result
        
        # If we're missing signal but have other fields, add a default signal
        if "signal" not in result and ("confidence" in result or "reasoning" in result):
            result["signal"] = "neutral"
            
        # If we're missing confidence but have other fields, add a default confidence
        if "confidence" not in result and ("signal" in result or "reasoning" in result):
            result["confidence"] = 0.5
            
        # If we're missing reasoning but have other fields, add a default reasoning
        if "reasoning" not in result and ("signal" in result or "confidence" in result):
            result["reasoning"] = "No detailed reasoning provided by the model."
        
        if result:
            return result
    except Exception as e:
        print(f"Error extracting key-value pairs: {e}")
    
    # If all else fails, return empty dict
    return {}

def extract_portfolio_decisions(content: str) -> dict:
    """Extracts portfolio decisions from LLM response."""
    import re
    import json
    
    # First try to extract JSON using the standard method
    json_result = extract_json_from_response(content)
    if json_result and "decisions" in json_result:
        return json_result
    
    # Check if the content contains a decisions object directly
    try:
        # Look for a JSON object that might be the decisions
        decisions_pattern = r'"decisions"\s*:\s*(\{[^}]+\})'
        decisions_match = re.search(decisions_pattern, content, re.DOTALL)
        if decisions_match:
            decisions_json = decisions_match.group(1)
            # Fix common JSON formatting issues
            decisions_json = decisions_json.replace("'", '"')  # Replace single quotes with double quotes
            decisions_json = re.sub(r'(\w+):', r'"\1":', decisions_json)  # Add quotes to keys
            decisions_json = re.sub(r',\s*\}', '}', decisions_json)  # Remove trailing commas
            
            try:
                decisions = json.loads(decisions_json)
                return {"decisions": decisions}
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    
    # If that fails, try to parse the decisions directly
    decisions = {}
    
    # Look for ticker symbols and associated decisions using multiple patterns
    ticker_patterns = [
        r'([A-Z]{1,5}):\s*(?:Action|Decision)?\s*[:-]\s*([a-zA-Z]+)',
        r'(?:For|Ticker)\s+([A-Z]{1,5})(?:[,:]|\s+I|\s+we)?\s+(?:recommend|suggest|advise|decide|would|will)?\s+(?:to\s+)?([a-zA-Z\s]+)',
        r'([A-Z]{1,5})(?:\s+stock)?\s*:\s*([a-zA-Z]+)\s+(\d+\s+shares|with\s+confidence)',
        r'Decision\s+for\s+([A-Z]{1,5})(?:\s+is)?\s*:\s*([a-zA-Z]+)',
        r'"([A-Z]{1,5})"\s*:\s*\{\s*"action"\s*:\s*"([a-zA-Z]+)"',
        r'([A-Z]{1,5})\s*-\s*([a-zA-Z]+)\s+\d+\s+shares'
    ]
    
    all_ticker_matches = []
    for pattern in ticker_patterns:
        matches = re.findall(pattern, content)
        if matches:
            all_ticker_matches.extend(matches)
    
    # If no matches found using patterns, try to find ticker symbols and then look for actions nearby
    if not all_ticker_matches:
        ticker_symbols = re.findall(r'\b([A-Z]{1,5})\b', content)
        unique_tickers = set(ticker_symbols)
        
        for ticker in unique_tickers:
            # Find action within 100 characters after ticker mention
            ticker_context = re.search(rf'{ticker}(.{{1,100}})', content)
            if ticker_context:
                context = ticker_context.group(1)
                # Look for action words in the context
                action_match = re.search(r'\b(buy|sell|short|cover|hold|purchase|acquire|dispose|maintain)\b', context, re.IGNORECASE)
                if action_match:
                    all_ticker_matches.append((ticker, action_match.group(1)))
    
    # Process all found ticker-action pairs
    for ticker, action in all_ticker_matches:
        ticker = ticker.strip().upper()
        
        # Skip if this ticker is already processed
        if ticker in decisions:
            continue
            
        # Normalize action
        action = action.lower().strip()
        action_mapping = {
            "purchase": "buy",
            "acquire": "buy",
            "long": "buy",
            "add": "buy",
            "increase": "buy",
            "accumulate": "buy",
            "invest": "buy",
            "dispose": "sell",
            "reduce": "sell",
            "exit": "sell",
            "liquidate": "sell",
            "decrease": "sell",
            "divest": "sell",
            "short sell": "short",
            "short position": "short",
            "go short": "short",
            "exit short": "cover",
            "cover short": "cover",
            "close short": "cover",
            "maintain": "hold",
            "keep": "hold",
            "no action": "hold",
            "wait": "hold",
            "observe": "hold",
            "monitor": "hold",
            "neutral": "hold"
        }
        normalized_action = action_mapping.get(action, action)
        if normalized_action not in ["buy", "sell", "short", "cover", "hold"]:
            normalized_action = "hold"  # Default to hold if invalid
        
        # Look for quantity using multiple patterns
        quantity_patterns = [
            rf'{ticker}.*?(?:quantity|shares|amount).*?(\d+)',
            rf'(\d+)\s+shares\s+of\s+{ticker}',
            rf'{ticker}.*?{normalized_action}\s+(\d+)',
            rf'{normalized_action}\s+(\d+)\s+(?:shares|stocks).*?{ticker}'
        ]
        
        quantity = 0
        for pattern in quantity_patterns:
            quantity_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if quantity_match:
                try:
                    quantity = int(quantity_match.group(1))
                    break
                except ValueError:
                    continue
        
        # Look for confidence using multiple patterns
        confidence_patterns = [
            rf'{ticker}.*?confidence.*?([\d\.]+)',
            rf'confidence.*?{ticker}.*?([\d\.]+)',
            rf'confidence\s+(?:of|is|at)\s+([\d\.]+).*?{ticker}',
            rf'{ticker}.*?confidence\s+(?:of|is|at)\s+([\d\.]+)'
        ]
        
        confidence = 50.0
        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if confidence_match:
                try:
                    confidence_value = float(confidence_match.group(1))
                    # Handle percentage values
                    if confidence_value > 0 and confidence_value <= 1:
                        confidence = confidence_value * 100
                    elif confidence_value > 1 and confidence_value <= 100:
                        confidence = confidence_value
                    else:
                        confidence = 50.0
                    break
                except ValueError:
                    continue
        
        # Look for reasoning using multiple patterns
        reasoning_patterns = [
            rf'{ticker}.*?(?:reasoning|rationale).*?[:]\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            rf'(?:reasoning|rationale)\s+for\s+{ticker}.*?[:]\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            rf'{ticker}.*?(?:because|due to|based on).*?[:]\s*(.+?)(?=\n\n|\n[A-Z]|$)'
        ]
        
        reasoning = "No reasoning provided"
        for pattern in reasoning_patterns:
            reasoning_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                break
        
        # If no specific reasoning found, try to extract a general reasoning from nearby text
        if reasoning == "No reasoning provided":
            ticker_pos = content.find(ticker)
            if ticker_pos != -1:
                # Extract 200 characters after the ticker mention
                context_after = content[ticker_pos:ticker_pos+200]
                # Remove any JSON or code blocks
                clean_context = re.sub(r'```.*?```', '', context_after, flags=re.DOTALL)
                clean_context = re.sub(r'\{.*?\}', '', clean_context, flags=re.DOTALL)
                # Use this as reasoning
                if clean_context:
                    reasoning = clean_context.strip()
        
        decisions[ticker] = {
            "action": normalized_action,
            "quantity": quantity,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    # If we found any decisions, return them
    if decisions:
        return {"decisions": decisions}
    
    # Last resort: try to extract any structured data that looks like a decision
    try:
        # Look for any JSON-like structures
        json_like_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_like_pattern, content)
        
        for json_str in json_matches:
            try:
                # Fix common JSON formatting issues
                fixed_json = json_str.replace("'", '"')  # Replace single quotes with double quotes
                fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)  # Add quotes to keys
                fixed_json = re.sub(r',\s*\}', '}', fixed_json)  # Remove trailing commas
                
                obj = json.loads(fixed_json)
                
                # Check if this looks like a decision object
                if "action" in obj and isinstance(obj["action"], str):
                    action = obj["action"].lower()
                    if action in ["buy", "sell", "short", "cover", "hold"]:
                        # Try to find a ticker for this decision
                        ticker_match = re.search(r'([A-Z]{1,5})', json_str)
                        if ticker_match:
                            ticker = ticker_match.group(1)
                            decisions[ticker] = {
                                "action": action,
                                "quantity": obj.get("quantity", 0),
                                "confidence": obj.get("confidence", 50.0),
                                "reasoning": obj.get("reasoning", "Extracted from JSON-like structure")
                            }
            except Exception:
                continue
    except Exception:
        pass
    
    # If all else fails, return an empty decisions dict
    return {"decisions": decisions}

def extract_investment_signal(content: str) -> dict:
    """Extracts investment signals from Warren Buffett or Charlie Munger responses."""
    import re
    
    # First try to extract JSON using the standard method
    json_result = extract_json_from_response(content)
    if json_result and "signal" in json_result and "confidence" in json_result and "reasoning" in json_result:
        # Ensure the signal is normalized even when extracted from JSON
        signal_value = json_result["signal"].lower().strip()
        signal_mapping = {
            # Neutral signals
            "cautious": "neutral",
            "hold": "neutral",
            "wait": "neutral",
            "observe": "neutral",
            "monitor": "neutral",
            "consider": "neutral",
            "wait and see": "neutral",
            "on the fence": "neutral",
            "mixed": "neutral",
            "balanced": "neutral",
            "uncertain": "neutral",
            "undecided": "neutral",
            "neither bullish nor bearish": "neutral",
            "cautious_bearish": "bearish",
            "cautious_bullish": "bullish",
            
            # Bearish signals
            "cautious bearish": "bearish",
            "slightly bearish": "bearish",
            "moderately bearish": "bearish",
            "strongly bearish": "bearish",
            "very bearish": "bearish",
            "sell": "bearish",
            "avoid": "bearish",
            "negative": "bearish",
            "not invest in": "bearish",
            "overvalued": "bearish",
            "reduce position": "bearish",
            "exit position": "bearish",
            "take profits": "bearish",
            
            # Bullish signals
            "cautious bullish": "bullish",
            "slightly bullish": "bullish",
            "moderately bullish": "bullish",
            "strongly bullish": "bullish",
            "very bullish": "bullish",
            "buy": "bullish",
            "invest in": "bullish",
            "positive": "bullish",
            "undervalued": "bullish",
            "increase position": "bullish",
            "add to position": "bullish",
            "accumulate": "bullish"
        }
        json_result["signal"] = signal_mapping.get(signal_value, signal_value)
        
        # Ensure signal is one of the allowed values
        if json_result["signal"] not in ["bullish", "bearish", "neutral"]:
            json_result["signal"] = "neutral"  # Default to neutral if invalid
            
        return json_result
    
    # If that fails, try to parse the signal directly
    signal_patterns = [
        r"(?:signal|recommendation|decision|investment|stance|position|outlook|approach)s?:?\s*([a-zA-Z\s]+)",
        r"I would (buy|sell|hold|avoid|invest in|not invest in|consider|recommend|suggest)(?:\s+\w+){0,5}\s+stock",
        r"(bullish|bearish|neutral)(?:\s+\w+){0,3}\s+(?:outlook|stance|position|view)",
        r"(buy|sell|hold)(?:\s+\w+){0,3}\s+(?:recommendation|advice|suggestion)",
        r"My (?:analysis|conclusion|recommendation) is (?:to )?(buy|sell|hold|bullish|bearish|neutral)"
    ]
    
    signal_value = None
    for pattern in signal_patterns:
        signal_match = re.search(pattern, content, re.IGNORECASE)
        if signal_match:
            signal_value = signal_match.group(1).lower().strip()
            break
    
    confidence_patterns = [
        r"confidence:?\s*([\d\.]+)",
        r"confidence(?:\s+\w+){0,3}\s+(?:of|is|at)\s+([\d\.]+)",
        r"([\d\.]+)%?\s+(?:confidence|certain|sure)",
        r"confidence level:?\s*([\d\.]+)"
    ]
    
    confidence_value = None
    for pattern in confidence_patterns:
        confidence_match = re.search(pattern, content, re.IGNORECASE)
        if confidence_match:
            confidence_value = confidence_match.group(1)
            break
    
    reasoning_patterns = [
        r"(?:reasoning|rationale|analysis|explanation|justification):?\s*(.+?)(?=\n\n|\n[A-Z]|$)",
        r"(?:because|due to|based on)\s+(.+?)(?=\n\n|\n[A-Z]|$)",
        r"(?:my|the) (?:reasoning|rationale|analysis) (?:is|includes|suggests)\s+(.+?)(?=\n\n|\n[A-Z]|$)"
    ]
    
    reasoning_value = None
    for pattern in reasoning_patterns:
        reasoning_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning_value = reasoning_match.group(1).strip()
            break
    
    result = {}
    
    # Process signal
    if signal_value:
        # Map common non-standard values to standard ones
        signal_mapping = {
            # Neutral signals
            "cautious": "neutral",
            "hold": "neutral",
            "wait": "neutral",
            "observe": "neutral",
            "monitor": "neutral",
            "consider": "neutral",
            "wait and see": "neutral",
            "on the fence": "neutral",
            "mixed": "neutral",
            "balanced": "neutral",
            "uncertain": "neutral",
            "undecided": "neutral",
            "neither bullish nor bearish": "neutral",
            
            # Bearish signals
            "cautious bearish": "bearish",
            "slightly bearish": "bearish",
            "moderately bearish": "bearish",
            "strongly bearish": "bearish",
            "very bearish": "bearish",
            "sell": "bearish",
            "avoid": "bearish",
            "negative": "bearish",
            "not invest in": "bearish",
            "overvalued": "bearish",
            "reduce position": "bearish",
            "exit position": "bearish",
            "take profits": "bearish",
            
            # Bullish signals
            "cautious bullish": "bullish",
            "slightly bullish": "bullish",
            "moderately bullish": "bullish",
            "strongly bullish": "bullish",
            "very bullish": "bullish",
            "buy": "bullish",
            "invest in": "bullish",
            "positive": "bullish",
            "undervalued": "bullish",
            "increase position": "bullish",
            "add to position": "bullish",
            "accumulate": "bullish"
        }
        result["signal"] = signal_mapping.get(signal_value, signal_value)
        # Ensure signal is one of the allowed values
        if result["signal"] not in ["bullish", "bearish", "neutral"]:
            result["signal"] = "neutral"  # Default to neutral if invalid
    else:
        # If no signal found, check for bullish/bearish/neutral keywords in the content
        bullish_keywords = [
            r'\b(buy|bullish|positive|upside|growth|undervalued|opportunity|attractive|strong|recommend|invest in)\b',
            r'\b(good|excellent|favorable|promising|potential|upward|increase|rise|gain|profit)\b'
        ]
        
        bearish_keywords = [
            r'\b(sell|bearish|negative|downside|decline|overvalued|avoid|unattractive|weak|not recommend|not invest)\b',
            r'\b(bad|poor|unfavorable|concerning|risk|downward|decrease|fall|loss|expensive)\b'
        ]
        
        bullish_count = 0
        for pattern in bullish_keywords:
            if re.search(pattern, content, re.IGNORECASE):
                bullish_count += 1
        
        bearish_count = 0
        for pattern in bearish_keywords:
            if re.search(pattern, content, re.IGNORECASE):
                bearish_count += 1
        
        if bullish_count > bearish_count:
            result["signal"] = "bullish"
        elif bearish_count > bullish_count:
            result["signal"] = "bearish"
        else:
            result["signal"] = "neutral"
    
    # Process confidence
    if confidence_value:
        try:
            confidence = float(confidence_value)
            # Handle percentage values
            if confidence > 1 and confidence <= 100:
                confidence = confidence / 100
            # Ensure confidence is between 0 and 1
            confidence = max(0, min(1, confidence))
            result["confidence"] = confidence
        except ValueError:
            result["confidence"] = 0.5  # Default confidence
    else:
        # If no confidence found, add a default
        result["confidence"] = 0.5
    
    # Process reasoning
    if reasoning_value:
        result["reasoning"] = reasoning_value
    else:
        # If no reasoning found, extract a portion of the content as reasoning
        # Remove any potential JSON or code blocks
        clean_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        clean_content = re.sub(r'\{.*?\}', '', clean_content, flags=re.DOTALL)
        
        content_lines = clean_content.strip().split('\n')
        if len(content_lines) > 2:
            # Skip the first line which might be a header or greeting
            result["reasoning"] = ' '.join(content_lines[1:]).strip()
        else:
            result["reasoning"] = clean_content.strip()
    
    # Ensure confidence is a number between 0 and 100 for the final output
    if "confidence" in result:
        if result["confidence"] <= 1:
            result["confidence"] = result["confidence"] * 100
    
    return result
