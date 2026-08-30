/// 분류 히스토리 로컬 저장 (sqflite).
library;

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

/// 기록이 바뀔 때마다 1씩 증가 — 기록 탭·통합검색이 listen 해 다시 읽는다.
/// (IndexedStack 으로 살아 있는 화면은 initState 를 다시 타지 않아, 결과 화면에서
/// 저장한 기록이 앱을 재시작하기 전까지 보이지 않던 문제의 해결책.)
final ValueNotifier<int> historyRevision = ValueNotifier(0);

void _bumpRevision() => historyRevision.value++;

class HistoryEntry {
  final int? id;
  final DateTime createdAt;
  final String predictedClass;
  final double confidence;
  final String imagePath;  // 로컬 파일 경로
  final String? uploadId;
  final String modelArch;

  HistoryEntry({
    this.id,
    required this.createdAt,
    required this.predictedClass,
    required this.confidence,
    required this.imagePath,
    this.uploadId,
    required this.modelArch,
  });

  Map<String, dynamic> toMap() => {
        if (id != null) 'id': id,
        'created_at': createdAt.toIso8601String(),
        'predicted_class': predictedClass,
        'confidence': confidence,
        'image_path': imagePath,
        'upload_id': uploadId,
        'model_arch': modelArch,
      };

  factory HistoryEntry.fromMap(Map<String, dynamic> m) => HistoryEntry(
        id: m['id'] as int?,
        createdAt: DateTime.parse(m['created_at'] as String),
        predictedClass: m['predicted_class'] as String,
        confidence: (m['confidence'] as num).toDouble(),
        imagePath: m['image_path'] as String,
        uploadId: m['upload_id'] as String?,
        modelArch: m['model_arch'] as String? ?? 'unknown',
      );
}


class HistoryRepository {
  static Database? _db;

  /// 보존 상한 — 초과분은 오래된 것부터 이미지 파일과 함께 삭제.
  /// (상한 없이는 장당 ~220KB 가 무한 누적 — 헤비유저 1년에 수백 MB)
  static const int kMaxEntries = 100;

  /// 열린 DB 핸들을 닫고 캐시를 비운다 — 다음 호출에서 다시 연다.
  /// (테스트가 임시 디렉토리를 바꿀 때 필요. 앱 런타임에서는 쓰지 않는다.)
  static Future<void> reset() async {
    await _db?.close();
    _db = null;
  }

  static Future<Database> _open() async {
    if (_db != null) return _db!;
    final docsDir = await getApplicationDocumentsDirectory();
    final dbPath = p.join(docsDir.path, 'history.db');
    _db = await openDatabase(
      dbPath,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          create table history (
            id integer primary key autoincrement,
            created_at text not null,
            predicted_class text not null,
            confidence real not null,
            image_path text not null,
            upload_id text,
            model_arch text
          )
        ''');
        await db.execute('create index history_created_at_idx on history(created_at desc)');
      },
    );
    return _db!;
  }

  /// 결과 이미지를 documents 디렉토리에 복사하고 메타와 함께 저장.
  Future<HistoryEntry> save({
    required File sourceImage,
    required String predictedClass,
    required double confidence,
    required String? uploadId,
    required String modelArch,
    DateTime? createdAt, // 생략 시 지금 — 개발용 더미 시드에서만 지정
  }) async {
    final docsDir = await getApplicationDocumentsDirectory();
    final imagesDir = Directory(p.join(docsDir.path, 'history_images'));
    if (!imagesDir.existsSync()) imagesDir.createSync(recursive: true);

    final filename = '${DateTime.now().millisecondsSinceEpoch}_${p.basename(sourceImage.path)}';
    final dest = File(p.join(imagesDir.path, filename));
    await sourceImage.copy(dest.path);

    final entry = HistoryEntry(
      createdAt: createdAt ?? DateTime.now(),
      predictedClass: predictedClass,
      confidence: confidence,
      imagePath: dest.path,
      uploadId: uploadId,
      modelArch: modelArch,
    );

    final db = await _open();
    final id = await db.insert('history', entry.toMap());
    await _pruneOverCap(db);
    _bumpRevision();
    return HistoryEntry(
      id: id,
      createdAt: entry.createdAt,
      predictedClass: entry.predictedClass,
      confidence: entry.confidence,
      imagePath: entry.imagePath,
      uploadId: entry.uploadId,
      modelArch: entry.modelArch,
    );
  }

  /// 상한 초과분(오래된 순) 행 + 이미지 파일 삭제.
  Future<void> _pruneOverCap(Database db) async {
    final over = await db.query(
      'history',
      columns: ['id', 'image_path'],
      orderBy: 'created_at desc',
      offset: kMaxEntries,
      limit: 1000,
    );
    for (final row in over) {
      final path = row['image_path'] as String?;
      if (path != null) {
        try {
          final f = File(path);
          if (f.existsSync()) f.deleteSync();
        } catch (_) {
          // 이미지 파일 삭제는 best-effort — 실패해도 DB 행은 지운다
        }
      }
      await db.delete('history', where: 'id = ?', whereArgs: [row['id']]);
    }
  }

  Future<List<HistoryEntry>> recent({int limit = 100}) async {
    final db = await _open();
    final rows = await db.query(
      'history',
      orderBy: 'created_at desc',
      limit: limit,
    );
    return rows.map(HistoryEntry.fromMap).toList();
  }

  Future<int> count() async {
    final db = await _open();
    final result = await db.rawQuery('select count(*) as c from history');
    return result.first['c'] as int? ?? 0;
  }

  Future<void> delete(int id) async {
    final db = await _open();
    // 이미지 파일도 함께 삭제 (행만 지우면 파일이 고아로 누적)
    final rows = await db.query('history',
        columns: ['image_path'], where: 'id = ?', whereArgs: [id]);
    for (final row in rows) {
      final path = row['image_path'] as String?;
      if (path != null) {
        try {
          final f = File(path);
          if (f.existsSync()) f.deleteSync();
        } catch (_) {
          // 이미지 파일 삭제는 best-effort — 실패해도 DB 행은 지운다
        }
      }
    }
    await db.delete('history', where: 'id = ?', whereArgs: [id]);
    _bumpRevision();
  }

  Future<void> clear() async {
    final db = await _open();
    await db.delete('history');
    // 이미지 폴더 통째로 정리
    try {
      final docsDir = await getApplicationDocumentsDirectory();
      final imagesDir = Directory(p.join(docsDir.path, 'history_images'));
      if (imagesDir.existsSync()) imagesDir.deleteSync(recursive: true);
    } catch (_) {
      // 폴더 정리는 best-effort — DB 는 이미 비웠다
    }
    _bumpRevision();
  }
}
