import numpy as np
import json

def main():
    # 1. 파일 경로 설정
    attack_file_path = '/home/student/log_project/attack/attack_drained_data.txt' 
    
    # 🌟 [수정 1] 결과물을 log_project 메인 폴더에 저장하도록 절대 경로로 변경
    save_file_name = '/home/student/log_project/attack_numeric_logs_1d.npy'
    
    # 2. 🌟 기준 번역기(정상 단어장) 로드
    try:
        # 🌟 [수정 2] id_map.json의 절대 경로 입력
        with open('/home/student/log_project/id_map.json', 'r', encoding='utf-8') as f:
            id_map = json.load(f)
        print(f"[*] 기준 단어장 로드 완료 (현재 단어 수: {len(id_map)}개)")
    except FileNotFoundError:
        print("[!] 에러: id_map.json이 없습니다. 정상 로그 추출을 먼저 해주세요!")
        return

    # 3. 공격 로그 텍스트 읽기
    try:
        with open(attack_file_path, 'r', encoding='utf-8') as f:
            attack_logs = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"[!] 에러: 공격 파일을 찾을 수 없습니다: {attack_file_path}")
        return

    # 4. 공격 로그를 숫자로 번역 (모르는 단어 처리 로직 포함)
    numeric_logs = []
    new_attack_patterns = 0
    
    for log in attack_logs:
        if log not in id_map:
            id_map[log] = len(id_map)
            new_attack_patterns += 1
            
        numeric_logs.append(id_map[log])

    # 5. 업데이트된 단어장을 다시 덮어쓰기 저장
    if new_attack_patterns > 0:
        # 🌟 [수정 3] 업데이트된 내용을 원래 위치의 id_map.json에 덮어쓰도록 절대 경로 입력
        with open('/home/student/log_project/id_map.json', 'w', encoding='utf-8') as f:
            json.dump(id_map, f, ensure_ascii=False, indent=4)
        print(f"[!] 새로운 공격 패턴 {new_attack_patterns}개가 단어장에 신규 등록되었습니다!")

    # 6. 공격 1차원 데이터 저장
    np.save(save_file_name, np.array(numeric_logs))
    print(f"[+] 완료! 공격 테스트용 데이터 '{save_file_name}' (총 {len(numeric_logs)}줄) 생성 완료!")

if __name__ == "__main__":
    main()
