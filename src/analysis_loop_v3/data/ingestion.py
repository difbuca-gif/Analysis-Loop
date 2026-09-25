"""입력 데이터 준비.

일반 파일은 그대로 사용한다. 디렉터리는 데이터셋별 관계를 모른 채 자동 조인하지
않는다. 잘못된 many-to-many 조인은 금액과 건수를 조용히 부풀리기 때문이다. 현재는
명시적으로 식별 가능한 Olist 스키마만 주문 단위 분석 뷰로 변환한다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from . import PreparedDataset

_OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "translations": "product_category_name_translation.csv",
}


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_fingerprint(paths: list[Path]) -> str:
    """mtime이 아니라 실제 파일 내용으로 준비 데이터 버전을 식별한다."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(_file_sha256(path).encode())
    return digest.hexdigest()[:20]


def _first_mode(series: pd.Series) -> Any:
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else None


def _duration_days(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.total_seconds() / 86_400


def _prepare_olist(source: Path, output_dir: Path) -> PreparedDataset:
    files = {name: source / filename for name, filename in _OLIST_FILES.items()}
    missing = [path.name for path in files.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Olist 데이터셋에 필요한 파일이 없다: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _source_fingerprint(list(files.values()))
    output = output_dir / f"olist-order-view-{fingerprint}.parquet"
    if output.is_file():
        frame = pd.read_parquet(output, columns=["order_id"])
        return PreparedDataset(
            path=output,
            summary={
                "source": str(source), "source_kind": "olist_directory",
                "source_fingerprint": fingerprint,
                "grain": "order", "rows": len(frame), "cached_preparation": True,
            },
        )

    order_dates = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    orders = pd.read_csv(files["orders"], parse_dates=order_dates)
    customers = pd.read_csv(files["customers"], dtype={"customer_zip_code_prefix": "string"})
    items = pd.read_csv(files["items"])
    products = pd.read_csv(files["products"])
    sellers = pd.read_csv(files["sellers"], dtype={"seller_zip_code_prefix": "string"})
    payments = pd.read_csv(files["payments"])
    reviews = pd.read_csv(files["reviews"])
    translations = pd.read_csv(files["translations"], encoding="utf-8-sig")

    item_detail = (
        items.merge(products, on="product_id", how="left", validate="many_to_one")
        .merge(translations, on="product_category_name", how="left", validate="many_to_one")
        .merge(
            sellers[["seller_id", "seller_state"]],
            on="seller_id", how="left", validate="many_to_one",
        )
    )
    item_detail["category"] = item_detail["product_category_name_english"].fillna(
        item_detail["product_category_name"]
    )
    item_agg = item_detail.groupby("order_id", sort=False).agg(
        item_count=("order_item_id", "count"),
        product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        item_price_total=("price", "sum"),
        item_price_mean=("price", "mean"),
        freight_value_total=("freight_value", "sum"),
        category_count=("category", "nunique"),
        product_weight_g_mean=("product_weight_g", "mean"),
        product_photos_qty_mean=("product_photos_qty", "mean"),
    ).reset_index()
    item_modes = item_detail.groupby("order_id", sort=False).agg(
        primary_category=("category", _first_mode),
        primary_seller_state=("seller_state", _first_mode),
    ).reset_index()
    item_agg = item_agg.merge(item_modes, on="order_id", validate="one_to_one")

    payment_agg = payments.groupby("order_id", sort=False).agg(
        payment_value_total=("payment_value", "sum"),
        payment_count=("payment_sequential", "count"),
        payment_type_count=("payment_type", "nunique"),
        payment_installments_max=("payment_installments", "max"),
    ).reset_index()
    payment_modes = payments.groupby("order_id", sort=False).agg(
        primary_payment_type=("payment_type", _first_mode),
    ).reset_index()
    payment_agg = payment_agg.merge(payment_modes, on="order_id", validate="one_to_one")

    review_message = reviews["review_comment_message"].fillna("").astype("string").str.strip()
    reviews["has_review_comment"] = review_message.ne("")
    review_agg = reviews.groupby("order_id", sort=False).agg(
        review_score_mean=("review_score", "mean"),
        review_score_min=("review_score", "min"),
        review_count=("review_id", "count"),
        has_review_comment=("has_review_comment", "max"),
    ).reset_index()

    view = orders.merge(customers, on="customer_id", how="left", validate="many_to_one")
    for aggregate in (item_agg, payment_agg, review_agg):
        view = view.merge(aggregate, on="order_id", how="left", validate="one_to_one")

    purchase = view["order_purchase_timestamp"]
    delivered = view["order_delivered_customer_date"]
    estimated = view["order_estimated_delivery_date"]
    view["purchase_year"] = purchase.dt.year.astype("Int16")
    view["purchase_month"] = purchase.dt.month.astype("Int8")
    view["purchase_dayofweek"] = purchase.dt.dayofweek.astype("Int8")
    view["purchase_hour"] = purchase.dt.hour.astype("Int8")
    view["approval_hours"] = (
        _duration_days(view["order_approved_at"], purchase) * 24
    )
    view["delivery_days"] = _duration_days(delivered, purchase)
    view["estimated_delivery_days"] = _duration_days(estimated, purchase)
    view["delivery_delay_days"] = _duration_days(delivered, estimated)
    valid_delivery = delivered.notna() & estimated.notna()
    view["is_late_delivery"] = delivered.gt(estimated).where(valid_delivery).astype("Int8")

    if len(view) != len(orders) or not view["order_id"].is_unique:
        raise ValueError(
            "Olist 준비 후 주문 grain이 깨졌다 — many-to-many 조인 가능성을 확인하라"
        )

    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    view.to_parquet(temporary, index=False)
    temporary.replace(output)
    return PreparedDataset(
        path=output,
        summary={
            "source": str(source), "source_kind": "olist_directory",
            "source_fingerprint": fingerprint,
            "grain": "order", "rows": len(view), "columns": len(view.columns),
            "used_tables": list(_OLIST_FILES), "cached_preparation": False,
        },
    )


def prepare_dataset(source: Path | str, *, output_dir: Path | str) -> PreparedDataset:
    source = Path(source).resolve()
    if source.is_file():
        return PreparedDataset(
            path=source,
            summary={
                "source": str(source),
                "source_kind": "file",
                "source_sha256": _file_sha256(source),
            },
        )
    if not source.is_dir():
        raise FileNotFoundError(f"데이터 입력을 찾을 수 없다: {source}")

    present = {path.name for path in source.glob("*.csv")}
    required = set(_OLIST_FILES.values())
    if required <= present:
        return _prepare_olist(source, Path(output_dir))
    raise ValueError(
        "지원하지 않는 관계형 데이터 디렉터리다. 자동 조인은 중복 집계를 만들 수 있어 "
        "스키마별 어댑터 없이 실행하지 않는다."
    )
