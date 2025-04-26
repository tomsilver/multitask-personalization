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
    compressed_data = "eyJhbnN3ZXJzIjpbeyJvY2NsdXNpb24iOnsiciI6WyJmcm9udCJdLCJvIjpbImZyb250Il19LCJiaXRlX29yZGVyIjoiZnJlbmNoIGZyaWVzIGRpcHBlZCBpbiBrZXRjaHVwIiwicmVhZHlfc2lnbmFsIjoiYnV0dG9uIiwidmVyYmFsIjoiVHJ1ZSIsImxvb2tfZm9yd2FyZCI6IlllcyIsImJsb2NrX2ZvcndhcmQiOiJZZXMiLCJsb29rX2xlZnQiOm51bGwsImJsb2NrX2xlZnQiOm51bGwsInByZWZlcmVuY2VfcmF0aW5nIjoiNCJ9LHsib2NjbHVzaW9uIjp7InIiOlsiZnJvbnQiLCJsZWZ0Il0sIm8iOlsiZnJvbnQiLCJsZWZ0Il19LCJiaXRlX29yZGVyIjoiY2Fycm90IHN0aWNrcyBkaXBwZWQgaW4gaHVtbXVzIiwicmVhZHlfc2lnbmFsIjoiYnV0dG9uIiwidmVyYmFsIjoiRmFsc2UiLCJsb29rX2ZvcndhcmQiOiJZZXMiLCJibG9ja19mb3J3YXJkIjoiWWVzIiwibG9va19sZWZ0IjoiWWVzIiwiYmxvY2tfbGVmdCI6IlllcyIsInByZWZlcmVuY2VfcmF0aW5nIjoiNiJ9LHsib2NjbHVzaW9uIjp7InIiOlsiZnJvbnQiXSwibyI6W119LCJiaXRlX29yZGVyIjoicG90YXRvIHdlZGdlcyBkaXBwZWQgaW4ga2V0Y2h1cCIsInJlYWR5X3NpZ25hbCI6ImJ1dHRvbiIsInZlcmJhbCI6IlRydWUiLCJsb29rX2ZvcndhcmQiOiJZZXMiLCJibG9ja19mb3J3YXJkIjoiTm8iLCJsb29rX2xlZnQiOm51bGwsImJsb2NrX2xlZnQiOm51bGwsInByZWZlcmVuY2VfcmF0aW5nIjoiMSJ9LHsib2NjbHVzaW9uIjp7InIiOlsiZnJvbnQiXSwibyI6W119LCJiaXRlX29yZGVyIjoiY2VsZXJ5IHN0aWNrcyBkaXBwZWQgaW4gaHVtbXVzIiwicmVhZHlfc2lnbmFsIjoiYnV0dG9uIiwidmVyYmFsIjoiRmFsc2UiLCJsb29rX2ZvcndhcmQiOiJZZXMiLCJibG9ja19mb3J3YXJkIjoiTm8iLCJsb29rX2xlZnQiOm51bGwsImJsb2NrX2xlZnQiOm51bGwsInByZWZlcmVuY2VfcmF0aW5nIjoiMiJ9LHsib2NjbHVzaW9uIjp7InIiOlsiZnJvbnQiLCJsZWZ0Il0sIm8iOltdfSwiYml0ZV9vcmRlciI6InRhdGVyIHRvdHMgZGlwcGVkIGluIGtldGNodXAiLCJyZWFkeV9zaWduYWwiOiJidXR0b24iLCJ2ZXJiYWwiOiJGYWxzZSIsImxvb2tfZm9yd2FyZCI6IlllcyIsImJsb2NrX2ZvcndhcmQiOiJObyIsImxvb2tfbGVmdCI6IlllcyIsImJsb2NrX2xlZnQiOiJObyIsInByZWZlcmVuY2VfcmF0aW5nIjoiMiJ9XSwiaW50YWtlRGF0YSI6eyJuYW1lIjoiVG9tIiwiYWdlIjoiMzEiLCJnZW5kZXIiOiJNYWxlIiwicm9ib3RFeHBlcmllbmNlIjoiWWVzIiwiZmVkRXhwZXJpZW5jZSI6Ik5vIn19"
    # Decompress and print the state
    decompressed_state = decompress_state(compressed_data)
    print(json.dumps(decompressed_state, indent=2))
