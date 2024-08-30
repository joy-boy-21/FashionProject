import webview

def create_webview():
    # Assuming Flask is running on localhost:5000
    webview.create_window('VOUGE VANITY', 'http://127.0.0.1:5000')
    webview.start()

if __name__ == '__main__':
    create_webview()
