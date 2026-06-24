#!/usr/bin/env python3
"""
One-time setup script to derive Polymarket API credentials.

Prerequisites:
  pip install py-clob-client-v2

Usage:
  1. Set POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER_ADDRESS in your .env
  2. Run: python setup_credentials.py
  3. Copy the output into your .env file

This script connects to Polymarket's CLOB, derives your API key/secret/passphrase,
and prints them so you can paste them into .env.
"""
import os
import sys
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Installing python-dotenv...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv
    load_dotenv()


def main():
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
    signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "3"))

    if not private_key or private_key == "0x_your_private_key_here":
        print("ERROR: Set POLYMARKET_PRIVATE_KEY in your .env file first.")
        print("   Example: POLYMARKET_PRIVATE_KEY=0xabcdef123456...")
        sys.exit(1)

    if not funder_address or funder_address == "0x_your_deposit_wallet_address":
        print("ERROR: Set POLYMARKET_FUNDER_ADDRESS in your .env file first.")
        print("   Example: POLYMARKET_FUNDER_ADDRESS=0x1234567890abcdef...")
        sys.exit(1)

    print("=" * 60)
    print("  POLYMARKET CREDENTIAL DERIVATION")
    print("=" * 60)
    print(f"  Private Key: {private_key[:6]}...{private_key[-4:]}")
    print(f"  Funder:      {funder_address[:6]}...{funder_address[-4:]}")
    print(f"  Sig Type:    {signature_type}")
    print("=" * 60)
    print()

    try:
        from py_clob_client_v2 import ClobClient
        from py_clob_client_v2.clob_types import SignatureTypeV2
    except ImportError:
        print("Installing py-clob-client-v2...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "py-clob-client-v2"])
        from py_clob_client_v2 import ClobClient
        from py_clob_client_v2.clob_types import SignatureTypeV2

    # Map signature type number to enum
    sig_map = {
        0: SignatureTypeV2.EOA,
        1: SignatureTypeV2.POLY_PROXY,
        2: SignatureTypeV2.POLY_GNOSIS_SAFE,
        3: SignatureTypeV2.POLY_1271,
    }
    sig_enum = sig_map.get(signature_type, SignatureTypeV2.POLY_1271)

    print("Connecting to Polymarket CLOB...")
    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=private_key,
        signature_type=sig_enum,
        funder=funder_address,
    )

    print("Deriving API credentials...")
    creds = client.create_or_derive_api_key()

    api_key = creds.api_key
    api_secret = creds.api_secret
    api_passphrase = creds.api_passphrase

    print()
    print("=" * 60)
    print("  CREDENTIALS DERIVED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Add these lines to your .env file:")
    print()
    print(f"POLYMARKET_API_KEY={api_key}")
    print(f"POLYMARKET_API_SECRET={api_secret}")
    print(f"POLYMARKET_API_PASSPHRASE={api_passphrase}")
    print()
    print("=" * 60)

    # Also check balance
    try:
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        data = client.get_balance_allowance(params)
        balance = float(data.get("balance", 0)) / 1e6
        allowance = float(data.get("allowance", 0)) / 1e6
        print(f"  Wallet Balance:  ${balance:,.2f} pUSD")
        print(f"  Allowance:       ${allowance:,.2f} pUSD")
        if balance == 0:
            print()
            print("  ⚠️  WARNING: Your wallet has 0 pUSD!")
            print("     Deposit USDC to your Polygon wallet and convert to pUSD")
            print("     at polymarket.com before trading.")
        if allowance == 0:
            print()
            print("  ⚠️  WARNING: Token allowance is 0!")
            print("     You need to approve the Exchange contract to spend your pUSD.")
            print("     This is done automatically when you deposit via polymarket.com,")
            print("     or you can set it via the CLOB client.")
        print("=" * 60)
    except Exception as e:
        print(f"  (Could not check balance: {e})")
        print("=" * 60)


if __name__ == "__main__":
    main()