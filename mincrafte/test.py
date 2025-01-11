import json
from anvil import Region

def mca_to_json(mca_file_path, output_json_path):
    try:
        # .mca 파일 열기
        region = Region.from_file(mca_file_path)
        data = {}

        # 32x32 청크의 가능한 좌표를 순회
        for chunk_x in range(32):
            for chunk_z in range(32):
                try:
                    # 청크 데이터 가져오기
                    chunk = region.get_chunk(chunk_x, chunk_z)
                    if "Anvil" in chunk:
                        level = chunk["Anvil"]
                        chunk_data = {
                            "xPos": level.get("xPos", None),
                            "zPos": level.get("zPos", None),
                            "entities": level.get("Entities", []),
                            "blocks": level.get("Blocks", []),
                            "biomes": level.get("Biomes", []),
                        }
                        data[f"Chunk {chunk_x},{chunk_z}"] = chunk_data
                    else:
                        print(f"'Level' tag missing in chunk ({chunk_x}, {chunk_z}). Skipping.")
                except Exception as chunk_error:
                    # 청크가 없거나 문제가 있는 경우 예외 처리
                    print(f"Skipping chunk ({chunk_x}, {chunk_z}): {chunk_error}")

        # JSON 파일로 저장
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

        print(f"Conversion complete. JSON saved to: {output_json_path}")
    except Exception as e:
        print(f"Error occurred: {e}")


# 사용 예시
mca_file_path = "C:\\Users\\user\\AppData\\Roaming\\.minecraft\\saves\\afdsgnvb\\region\\r.0.0.mca"
output_json_path = "output.json"
mca_to_json(mca_file_path, output_json_path)
