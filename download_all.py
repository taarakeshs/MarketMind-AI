from src.universe import get_universe
from src.data import (
    download_universe_batch,
    combine_stock_data,
    save_combined_dataset,
)


def main():

    print("=" * 70)
    print("STOCK MARKET INTELLIGENCE SYSTEM")
    print("FULL HISTORICAL DATA DOWNLOAD")
    print("=" * 70)

    # -------------------------------------------------
    # Load complete universe
    # -------------------------------------------------

    tickers = get_universe(
        use_india=True,
        use_us=True,
    )

    print()
    print(
        f"Candidate stocks: {len(tickers)}"
    )

    # -------------------------------------------------
    # Download
    # -------------------------------------------------

    successful, failed = (
        download_universe_batch(
            tickers=tickers,
            period="3y",
            interval="1d",
            batch_size=25,
        )
    )

    # -------------------------------------------------
    # Save successful ticker list
    # -------------------------------------------------

    with open(
        "data/successful_tickers.txt",
        "w",
        encoding="utf-8",
    ) as file:

        for ticker in successful:
            file.write(
                ticker + "\n"
            )

    # -------------------------------------------------
    # Save failed ticker list
    # -------------------------------------------------

    with open(
        "data/failed_tickers.txt",
        "w",
        encoding="utf-8",
    ) as file:

        for ticker in failed:
            file.write(
                ticker + "\n"
            )

    # -------------------------------------------------
    # Combine
    # -------------------------------------------------

    if successful:

        combined = combine_stock_data(
            successful
        )

        path = save_combined_dataset(
            combined,
            "market_data.csv",
        )

        print()
        print(
            f"Combined dataset saved to:"
        )

        print(path)

        print()
        print(
            f"Total rows: {len(combined):,}"
        )

    print()
    print("=" * 70)
    print("FULL DOWNLOAD FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()