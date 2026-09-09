-- 012: user_uploads 촬영 메타 컬럼 (feature/api-upload-meta)
-- 적용은 사용자 승인 후 수동으로. 컬럼 미배포 상태에서도 서버는 기본 컬럼만으로
-- 기록을 계속한다 (uploads._remote_record 의 재시도 fail-open).
alter table public.user_uploads
  add column if not exists orientation        smallint,      -- 앱이 읽은 EXIF Orientation (1/3/6/8)
  add column if not exists capture_mode       text,          -- 'smart' | 'gallery'
  add column if not exists quality_blur       real,
  add column if not exists quality_brightness real,
  add column if not exists crop_applied       boolean,
  add column if not exists crop_box           text,          -- "x,y,w,h" (정규화 좌표)
  add column if not exists exif_orientation   smallint,      -- 서버가 이미지에서 읽은 태그
  add column if not exists tta_rotation       smallint,      -- 채택된 TTA 회전 (0/90/180/270)
  add column if not exists tap_x              real,          -- 탭-투-셀렉트 좌표 (0~1)
  add column if not exists tap_y              real;
