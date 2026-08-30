"""지연 생성 싱글톤 — 무거운 모델/클라이언트 로더의 공통 형태.

    @lazy_singleton
    def get_segmenter() -> Segmenter:
        return Segmenter()

    get_segmenter()          # 첫 호출 시 생성, 이후 캐시
    get_segmenter.reset()    # 폐기 → 다음 호출 시 재생성 (테스트 격리·모델 리로드)
    get_segmenter.instance   # 생성 없이 현재 인스턴스 조회 (None 가능)
"""
from __future__ import annotations

import functools
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class lazy_singleton(Generic[T]):  # noqa: N801 — 데코레이터라 소문자
    def __init__(self, factory: Callable[[], T]) -> None:
        functools.update_wrapper(self, factory)
        self._factory = factory
        self.instance: T | None = None

    def __call__(self) -> T:
        if self.instance is None:
            self.instance = self._factory()
        return self.instance

    def reset(self) -> None:
        self.instance = None
