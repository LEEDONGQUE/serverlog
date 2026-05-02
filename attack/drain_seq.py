import numpy as np
import json

def main():
    # 1. 파일 경로 설정 (정상 로그 추출할 땐 정상 파일, 공격 추출할 땐 공격 파일 경로 넣기)
    file_path = '/home/student/log_project/attack/attack_drained_data.txt' # 예시 경로
    save_file_name = 'attack_numeric_logs_1d.npy' # 저장할 파일 이름 (상황에 맞게 변경)
    
    print(f"[*] {file_path} 로딩 중...")

    # 2. 일반 텍스트 파일 읽기 (Pandas 삭제, 가볍게!)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            event_ids = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"[!] 에러: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
        return

    # 3. 문자열을 숫자로 매핑하기 (사전 만들기)
    unique_ids = sorted(list(set(event_ids)))
    id_map = {id_str: i for i, id_str in enumerate(unique_ids)}

    # 단어장 저장 (나중에 해석할 때 필요함)
    # 주의: 공격 로그 뽑을 때는 이거 주석처리 하거나 이름 바꿔서 정상로그 단어장 안 덮어씌워지게 조심!
    with open('id_map.json', 'w', encoding='utf-8') as f:
        json.dump(id_map, f, ensure_ascii=False, indent=4)

    # 문자열 -> 숫자로 변환
    numeric_logs = [id_map[log] for log in event_ids]
    
    print(f"[*] 총 로그 라인 수: {len(numeric_logs)}")
    print(f"[*] 고유 템플릿 개수(Vocab Size): {len(unique_ids)}")

    # 4. 🌟 [핵심] 자르지 않고 1차원 배열 그대로 저장!
    np.save(save_file_name, np.array(numeric_logs))
    print(f"[+] 완료! 1차원 원본 데이터 '{save_file_name}'이(가) 생성되었습니다.")

if __name__ == "__main__":
    main()
