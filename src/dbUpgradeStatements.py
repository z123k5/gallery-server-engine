dbUpgradeStatements = [
    {
        "toVersion": 1,
        "statements": [
            # 用户表
            """CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                password TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                active INTEGER DEFAULT 1 NOT NULL,
                is_audited INTEGER DEFAULT 0 NOT NULL,
                is_admin INTEGER DEFAULT 0 NOT NULL,
                token_expire DATETIME DEFAULT CURRENT_TIMESTAMP,
                upload_limit_a_day INTEGER DEFAULT -1 NOT NULL
            )
            """,
            ]
    },
    #  add new statements below for next database version when required
    {
        "toVersion": 2,
        "statements": [
            # 图片表
            """CREATE TABLE IF NOT EXISTS media(
                user_id INTEGER NOT NULL REFERENCES users(id),
                identifier TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL,
                processInfo INTEGER DEFAULT 0, -- 0:未处理 1:已处理 2:已分类
                feature BLOB DEFAULT 0
            )
            """,

            # 分类名称表
            """CREATE TABLE IF NOT EXISTS classes(
                user_id INTEGER NOT NULL REFERENCES users(id),
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                artificial INTEGER DEFAULT 0
            )
            """,

            # 图片分类表
            """CREATE TABLE IF NOT EXISTS media_classes (
                user_id INTEGER NOT NULL,
                media_id TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                artificial INTEGER DEFAULT 0,
                PRIMARY KEY(user_id, media_id),
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(media_id) REFERENCES media(identifier),
                FOREIGN KEY(class_id) REFERENCES classes(id)
            )
            """,

            # 默认添加一些分类
            "INSERT INTO classes(user_id, name, artificial) VALUES(1, '人物', 0);",
            "INSERT INTO classes(user_id, name, artificial) VALUES(1, '动物', 0);",
            "INSERT INTO classes(user_id, name, artificial) VALUES(1, '植物', 0);",
            "INSERT INTO classes(user_id, name, artificial) VALUES(1, '食物', 0);",
            "INSERT INTO classes(user_id, name, artificial) VALUES(1, '建筑', 0);",
            "INSERT INTO classes(user_id, name, artificial) VALUES(1, '家具', 0);",
            "INSERT INTO classes(user_id, name, artificial) VALUES(1, '交通工具', 0);",
            "INSERT INTO classes(user_id, name, artificial) VALUES(1, '电子产品', 0);",
            "INSERT INTO classes(user_id, name, artificial) VALUES(1, '服装', 0);",
            "INSERT INTO classes(user_id, name, artificial) VALUES(1, '乐器', 0);",
            "INSERT INTO classes(user_id, name, artificial) VALUES(1, '屏幕截图', 0);"
        ]
    },
    {
        "toVersion": 3,
        "statements": [
            # 日志表
            """CREATE TABLE IF NOT EXISTS logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                action INTEGER NOT NULL,
                valueBool TEXT,
                valueInt TEXT,
                valueStr TEXT,
                valueFloat TEXT,
                valueInfo TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """CREATE TABLE IF NOT EXISTS media_metadata (
                user_id INTEGER NOT NULL,
                media_id TEXT NOT NULL,
                exif_lat REAL,
                exif_lon REAL,
                exif_dev TEXT,
                location TEXT,
                PRIMARY KEY (user_id, media_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
                FOREIGN KEY (media_id) REFERENCES media(identifier) ON DELETE CASCADE ON UPDATE CASCADE
            );
            """
        ]
    },
]
