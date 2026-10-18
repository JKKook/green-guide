/// 한글 어절 단위 줄바꿈.
///
/// Flutter 는 한글을 글자 단위로 끊으므로("검색할 수 있/어요") 어절 안의
/// 글자 사이에 보이지 않는 단어 결합자(U+2060)를 넣어 공백에서만 줄이 바뀌게 한다
/// (웹의 `word-break: keep-all` 대응). 화면·낭독기·말줄임·`\n` 강제 줄바꿈에는
/// 영향이 없다.
///
/// 그리는 시점에만 적용한다 — 검색 매칭 등 원문 비교에 쓰는 데이터에는 넣지 않는다.
library;

final RegExp _word = RegExp(r'\S+');

String keepAll(String text) =>
    text.replaceAllMapped(_word, (m) => m[0]!.split('').join('⁠'));
