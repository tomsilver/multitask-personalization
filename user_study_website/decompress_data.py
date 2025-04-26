import json
import base64

def decompress_state(compressed_data):
    # Decode base64
    json_data = base64.b64decode(compressed_data).decode('utf-8')
    # Parse JSON
    minimal_answers = json.loads(json_data)
    
    # Convert back to full format (similar to the JavaScript decompressState function)
    full_answers = []
    for answer in minimal_answers:
        full_answer = {}
        for key, value in answer.items():
            if key == 'occlusion':
                # Special handling for occlusion data
                full_answer[key] = {
                    'relevant_pois': value['r'],
                    'occluded_pois': value['o']
                }
            else:
                # For other answers, restore the full format
                full_answer[key] = {
                    'value': value,
                    'metadata': None
                }
        full_answers.append(full_answer)
    
    return full_answers


if __name__ == "__main__":
    # Example compressed data (base64 encoded JSON)
    compressed_data = "W3siZmVlZGluZ19zaWRlIjoibGVmdCIsImJpdGVfb3JkZXIiOiJmcmVuY2ggZnJpZXMgZGlwcGVkIGluIGtldGNodXAiLCJyZWFkeV9zaWduYWwiOiJtb3V0aF9vcGVuIiwidmVyYmFsIjoiVHJ1ZSIsIm9jY2x1c2lvbiI6eyJyIjpbImZyb250Il0sIm8iOltdfSwibG9va19mb3J3YXJkIjoiWWVzIiwiYmxvY2tfZm9yd2FyZCI6Ik5vIiwibG9va19sZWZ0IjoiTm8iLCJibG9ja19sZWZ0IjoiTm8ifSx7ImZlZWRpbmdfc2lkZSI6ImxlZnQiLCJiaXRlX29yZGVyIjoiY2VsZXJ5IGRpcHBlZCBpbiByYW5jaCBkcmVzc2luZyB0aGVuIGFwcGxlIHNsaWNlcyBkaXBwZWQgaW4gcmFuY2ggZHJlc3NpbmciLCJyZWFkeV9zaWduYWwiOiJtb3V0aF9vcGVuIiwidmVyYmFsIjoiRmFsc2UiLCJvY2NsdXNpb24iOnsiciI6WyJmcm9udCIsImxlZnQiXSwibyI6W119LCJsb29rX2ZvcndhcmQiOiJZZXMiLCJibG9ja19mb3J3YXJkIjoiTm8iLCJsb29rX2xlZnQiOiJZZXMiLCJibG9ja19sZWZ0IjoiTm8ifSx7ImZlZWRpbmdfc2lkZSI6ImxlZnQiLCJiaXRlX29yZGVyIjoic3RlYWsgd2l0aG91dCBhbnkgZGlwcGluZyB0aGVuIHBvdGF0b2VzIHdpdGhvdXQgYW55IGRpcHBpbmciLCJyZWFkeV9zaWduYWwiOiJtb3V0aF9vcGVuIiwidmVyYmFsIjoiRmFsc2UiLCJvY2NsdXNpb24iOnsiciI6WyJmcm9udCJdLCJvIjpbXX0sImxvb2tfZm9yd2FyZCI6IlllcyIsImJsb2NrX2ZvcndhcmQiOiJObyIsImxvb2tfbGVmdCI6Ik5vIiwiYmxvY2tfbGVmdCI6Ik5vIn0seyJmZWVkaW5nX3NpZGUiOiJsZWZ0IiwiYml0ZV9vcmRlciI6InBlYXIgc2xpY2VzIGRpcHBlZCBpbiByYW5jaCBkcmVzc2luZyIsInJlYWR5X3NpZ25hbCI6Im1vdXRoX29wZW4iLCJ2ZXJiYWwiOiJUcnVlIiwib2NjbHVzaW9uIjp7InIiOlsiZnJvbnQiXSwibyI6W119LCJsb29rX2ZvcndhcmQiOiJZZXMiLCJibG9ja19mb3J3YXJkIjoiTm8iLCJsb29rX2xlZnQiOiJObyIsImJsb2NrX2xlZnQiOiJObyJ9LHsiZmVlZGluZ19zaWRlIjoibGVmdCIsImJpdGVfb3JkZXIiOiJjaGlja2VuIG51Z2dldHMgZGlwcGVkIGluIHJhbmNoIGRyZXNzaW5nIiwicmVhZHlfc2lnbmFsIjoibW91dGhfb3BlbiIsInZlcmJhbCI6IkZhbHNlIiwib2NjbHVzaW9uIjp7InIiOlsiZnJvbnQiLCJsZWZ0Il0sIm8iOltdfSwibG9va19mb3J3YXJkIjoiWWVzIiwiYmxvY2tfZm9yd2FyZCI6Ik5vIiwibG9va19sZWZ0IjoiWWVzIiwiYmxvY2tfbGVmdCI6Ik5vIn1d"
    # Decompress and print the state
    decompressed_state = decompress_state(compressed_data)
    print(json.dumps(decompressed_state, indent=2))
