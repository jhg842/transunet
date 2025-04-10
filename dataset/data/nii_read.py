import nibabel as nib

# 파일 경로 (자신의 .nii.gz 경로로 바꿔주세요)
file_path = "/home/jhg842/transunet/dataset/data/label0001.nii.gz"

# nii.gz 파일 불러오기
nii = nib.load(file_path)

# 데이터 배열로 변환 (numpy)
volume = nii.get_fdata()

# shape 확인
print(f"파일 이름: {file_path}")

print(f"데이터 형태 (shape): {volume.shape}")

volume = volume[:,:,110]
volume = volume.flatten()
for i in volume:
    if i > 12:
        print(i)