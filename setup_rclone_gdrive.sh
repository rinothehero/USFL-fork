#!/bin/bash
###############################################################################
# setup_rclone_gdrive.sh — GPU 서버에서 rclone + Google Drive 설정
#
# 각 GPU 서버에서 한 번만 실행하면 됩니다.
#
# Usage:
#   bash setup_rclone_gdrive.sh
###############################################################################

echo "=========================================="
echo "  rclone + Google Drive Setup"
echo "=========================================="
echo ""

# 1. Check if rclone is installed
if command -v rclone &>/dev/null; then
    echo "[OK] rclone is installed: $(rclone --version | head -1)"
else
    echo "[!] rclone is not installed. Installing..."
    echo ""

    # Install rclone (official script)
    curl https://rclone.org/install.sh | sudo bash

    if command -v rclone &>/dev/null; then
        echo ""
        echo "[OK] rclone installed: $(rclone --version | head -1)"
    else
        echo "[ERROR] rclone installation failed."
        echo "Manual install: https://rclone.org/install/"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "  Step 1: Configure Google Drive remote"
echo "=========================================="
echo ""
echo "Run the following command to set up Google Drive:"
echo ""
echo "  rclone config"
echo ""
echo "When prompted:"
echo "  1. Choose 'n' for new remote"
echo "  2. Name it: gdrive"
echo "  3. Choose 'Google Drive' (type number for 'drive')"
echo "  4. Leave client_id and client_secret blank"
echo "  5. Scope: 1 (Full access)"
echo "  6. Leave root_folder_id blank"
echo "  7. Leave service_account_file blank"
echo "  8. Auto config: No (since this is a headless server)"
echo "     → Copy the URL, paste in your local browser"
echo "     → Authorize → Copy the verification code back"
echo "  9. Shared drive: No"
echo "  10. Confirm: Yes"
echo ""
echo "Note: For headless servers, you'll need to authorize via browser"
echo "on your local machine and paste the token back."
echo ""

read -rp "Press Enter to start rclone config, or Ctrl+C to skip... "
rclone config

echo ""
echo "=========================================="
echo "  Step 2: Test the connection"
echo "=========================================="
echo ""

# List available remotes
echo "Available remotes:"
rclone listremotes
echo ""

read -rp "Enter the remote name to test (e.g., gdrive): " REMOTE_NAME
if [ -z "$REMOTE_NAME" ]; then
    echo "Skipped."
else
    echo ""
    echo "Testing: rclone lsd ${REMOTE_NAME}:"
    rclone lsd "${REMOTE_NAME}:" 2>&1 | head -10
    echo ""

    if [ $? -eq 0 ]; then
        echo "[OK] Google Drive connection successful!"
        echo ""
        echo "=========================================="
        echo "  Step 3: Update deploy_servers.json"
        echo "=========================================="
        echo ""
        echo "On your LOCAL machine, update deploy_servers.json:"
        echo ""
        echo "  \"gdrive_rclone_remote\": \"${REMOTE_NAME}:USFL-experiments\""
        echo ""
        echo "This will upload results to Google Drive at:"
        echo "  My Drive / USFL-experiments / ..."
        echo ""
    else
        echo "[ERROR] Connection failed. Run 'rclone config' again."
    fi
fi

echo "Done."
