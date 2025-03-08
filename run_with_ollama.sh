#!/bin/bash
# Script to run the AI Hedge Fund with Ollama as the default LLM

# Default values
MODEL="mixtral:8x7b"
TICKERS="AAPL,MSFT,NVDA"
SHOW_REASONING=false

# Help function
function show_help {
  echo "Usage: ./run_with_ollama.sh [OPTIONS]"
  echo ""
  echo "Run the AI Hedge Fund with Ollama as the default LLM provider"
  echo ""
  echo "Options:"
  echo "  -m, --model MODEL       Specify the Ollama model to use (default: mixtral:8x7b)"
  echo "  -t, --tickers TICKERS   Comma-separated list of tickers (default: AAPL,MSFT,NVDA)"
  echo "  -r, --show-reasoning    Show reasoning from each agent"
  echo "  -h, --help              Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./run_with_ollama.sh"
  echo "  ./run_with_ollama.sh --model llama3:70b"
  echo "  ./run_with_ollama.sh --tickers AAPL,MSFT,NVDA --show-reasoning"
  echo "  ./run_with_ollama.sh --tickers AAPL,MSFT,NVDA --model mixtral:8x7b"
  exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    -t|--tickers)
      TICKERS="$2"
      shift 2
      ;;
    -r|--show-reasoning)
      SHOW_REASONING=true
      shift
      ;;
    -h|--help)
      show_help
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      ;;
  esac
done

# Build the command
CMD="poetry run python src/main.py --tickers $TICKERS --model $MODEL --provider Ollama"

# Add show-reasoning flag if enabled
if [ "$SHOW_REASONING" = true ]; then
  CMD="$CMD --show-reasoning"
fi

# Print the command being run
echo "Running: $CMD"
echo ""

# Execute the command
eval "$CMD"
