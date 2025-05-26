# Flask admin route to reset the database from Excel
from flask import redirect, url_for

@app.route('/admin/reset_db')
def admin_reset_db():
    import os
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db_from_excel()
    return redirect(url_for('db_status'))
