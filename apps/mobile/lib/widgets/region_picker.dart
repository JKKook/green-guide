import 'package:flutter/material.dart';
import 'package:geocoding/geocoding.dart';
import 'package:geolocator/geolocator.dart';

import '../core/di/app_scope.dart';
import '../data/region_data.dart';
import '../theme/app_theme.dart';
import 'korea_map.dart';

/// GPS 지역 인식 결과 — region 이 null 이면 error 에 사용자 안내 문구.
class RegionLocateResult {
  final (String, String)? region;
  final String? error;
  const RegionLocateResult({this.region, this.error});
}

/// GPS → 역지오코딩 → 내장 행정구역 목록과 대조 (온보딩·지역 선택기 공용).
Future<RegionLocateResult> locateRegionFromGps() async {
  try {
    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      return const RegionLocateResult(
          error: '위치 권한이 없어요 — 아래 지도에서 선택해주세요');
    }
    final pos = await Geolocator.getCurrentPosition(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.low, // 시군구 단위면 충분
        timeLimit: Duration(seconds: 10),
      ),
    );
    await setLocaleIdentifier('ko_KR'); // 행정구역명을 한국어로
    final placemarks =
        await placemarkFromCoordinates(pos.latitude, pos.longitude);
    final match = _matchRegion(placemarks);
    if (match == null) {
      return const RegionLocateResult(
          error: '지역을 인식하지 못했어요 — 아래 지도에서 선택해주세요');
    }
    return RegionLocateResult(region: match);
  } catch (_) {
    return const RegionLocateResult(
        error: '위치를 가져오지 못했어요 — 아래 지도에서 선택해주세요');
  }
}

/// 역지오코딩 결과를 내장 행정구역 목록과 대조 (표기 흔들림 방어).
(String, String)? _matchRegion(List<Placemark> placemarks) {
  for (final pm in placemarks) {
    final admin = pm.administrativeArea ?? '';
    final candidates = [
      pm.subAdministrativeArea,
      pm.locality,
      pm.subLocality,
    ].whereType<String>().where((s) => s.isNotEmpty);
    // 시도 매칭 — '서울' ↔ '서울특별시' 접두 흔들림 허용
    String? sido;
    for (final s in kKoreaRegions.keys) {
      if (s == admin || s.startsWith(admin) || admin.startsWith(s)) {
        sido = s;
        break;
      }
    }
    if (sido == null) continue;
    for (final c in candidates) {
      for (final g in kKoreaRegions[sido]!) {
        if (g == c || c.startsWith(g) || g.startsWith(c)) return (sido, g);
      }
    }
    // 세종처럼 시군구가 단일인 경우
    if (kKoreaRegions[sido]!.length == 1) {
      return (sido, kKoreaRegions[sido]!.first);
    }
  }
  return null;
}

/// 지역 선택 바텀시트.
///
/// 주 경로: 📍 내 위치로 설정 (GPS + 역지오코딩, 1탭)
/// 수동 경로: 대한민국 지도에서 시도 탭 (Level 1) → 시군구 리스트 (Level 2)
/// — 지자체 조례별로 배출 요일·시간·방법이 달라 선택 지역 기준으로 안내를 매핑.
/// 선택 완료 시 (sido, sigungu) 반환, 스킵이면 null.
Future<(String, String)?> showRegionPicker(BuildContext context) {
  return showModalBottomSheet<(String, String)>(
    context: context,
    isScrollControlled: true,
    useSafeArea: true,
    builder: (_) => const _RegionPickerSheet(),
  );
}

class _RegionPickerSheet extends StatefulWidget {
  const _RegionPickerSheet();

  @override
  State<_RegionPickerSheet> createState() => _RegionPickerSheetState();
}

class _RegionPickerSheetState extends State<_RegionPickerSheet> {
  String? _sido;         // Level 2 진입 상태
  bool _locating = false;
  String? _gpsError;

  /// GPS → 역지오코딩 → 시도/시군구 매칭 → 저장.
  Future<void> _useMyLocation() async {
    setState(() {
      _locating = true;
      _gpsError = null;
    });
    final result = await locateRegionFromGps();
    if (!mounted) return;
    if (result.region == null) {
      setState(() {
        _locating = false;
        _gpsError = result.error;
      });
      return;
    }
    await AppScope.settings.setRegion(result.region!.$1, result.region!.$2);
    if (mounted) Navigator.of(context).pop(result.region);
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.85,
      maxChildSize: 0.95,
      builder: (context, scroll) => Column(
        children: [
          const SizedBox(height: kSpaceM),
          Container(
            width: 36, height: 4,
            decoration: BoxDecoration(
              color: cs.outlineVariant,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(kSpaceL, kSpaceL, kSpaceL, kSpaceS),
            child: Row(
              children: [
                if (_sido != null)
                  IconButton(
                    icon: const Icon(Icons.arrow_back),
                    onPressed: () => setState(() => _sido = null),
                  ),
                Icon(Icons.place_outlined, color: cs.primary),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _sido ?? '어느 지역에 사시나요?',
                        style: Theme.of(context)
                            .textTheme
                            .titleMedium
                            ?.copyWith(fontWeight: FontWeight.w700),
                      ),
                      Text(
                        _sido == null
                            ? '지역마다 분리배출 기준이 조금씩 달라요'
                            : '시·군·구를 선택해주세요',
                        style: Theme.of(context)
                            .textTheme
                            .bodySmall
                            ?.copyWith(color: cs.onSurfaceVariant),
                      ),
                    ],
                  ),
                ),
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('나중에'),
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          Expanded(
            child: _sido == null
                ? _buildLevel1(scroll, cs)
                : _buildLevel2(scroll),
          ),
        ],
      ),
    );
  }

  /// Level 1 — GPS 버튼 + 대한민국 지도 (시도 선택).
  Widget _buildLevel1(ScrollController scroll, ColorScheme cs) {
    return ListView(
      controller: scroll,
      padding: const EdgeInsets.all(kSpaceL),
      children: [
        FilledButton.icon(
          onPressed: _locating ? null : _useMyLocation,
          icon: _locating
              ? const SizedBox(
                  width: 16, height: 16,
                  child: CircularProgressIndicator(strokeWidth: 2))
              : const Icon(Icons.my_location),
          label: Text(_locating ? '위치 확인 중...' : '내 위치로 설정'),
        ),
        if (_gpsError != null) ...[
          const SizedBox(height: kSpaceS),
          Text(
            _gpsError!,
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 12.5, color: cs.error),
          ),
        ],
        const SizedBox(height: kSpaceM),
        Row(
          children: [
            const Expanded(child: Divider()),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: kSpaceM),
              child: Text('또는 지도에서 선택',
                  style: TextStyle(
                      fontSize: 12, color: cs.onSurfaceVariant)),
            ),
            const Expanded(child: Divider()),
          ],
        ),
        const SizedBox(height: kSpaceS),
        // 핀치 줌/팬 — 수도권처럼 시도가 밀집된 영역 확대 선택용
        ClipRRect(
          borderRadius: BorderRadius.circular(kRadiusLarge),
          child: InteractiveViewer(
            maxScale: 6,
            child: KoreaMap(onSelect: (sido) => setState(() => _sido = sido)),
          ),
        ),
        const SizedBox(height: 6),
        Center(
          child: Text(
            '두 손가락으로 확대할 수 있어요',
            style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
          ),
        ),
      ],
    );
  }

  /// Level 2 — 선택 시도의 시군구 리스트.
  Widget _buildLevel2(ScrollController scroll) {
    final items = kKoreaRegions[_sido]!;
    return ListView.builder(
      controller: scroll,
      itemCount: items.length,
      itemBuilder: (context, i) {
        final name = items[i];
        return ListTile(
          title: Text(name),
          onTap: () async {
            await AppScope.settings.setRegion(_sido!, name);
            if (context.mounted) {
              Navigator.of(context).pop((_sido!, name));
            }
          },
        );
      },
    );
  }
}
