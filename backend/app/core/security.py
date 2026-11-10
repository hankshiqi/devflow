# 学习导读｜仓库访问 Token 的加密与解密封装。
# 数据库保存密文，实际调用 GitHub 时再解密；这不是用户登录或完整的身份认证模块。
from cryptography.fernet import Fernet, InvalidToken


# 从配置 secret 派生固定长度 Fernet 密钥，用于本地演示环境的 Token 加密。
def _derive_demo_key(secret: str) -> bytes:
    # Fernet 需要 urlsafe base64 编码的 32 字节密钥。
    # 这里的确定性本地开发派生方式适合课程项目，不适合生产环境。
    import base64
    import hashlib

    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())


# 用派生密钥把明文访问令牌加密后保存；数据库中不应存明文 Token。
def encrypt_token(token: str, secret: str) -> str:
    return Fernet(_derive_demo_key(secret)).encrypt(token.encode("utf-8")).decode("utf-8")


# 解密数据库中保存的访问令牌，供 GitHub 客户端发起认证请求。
def decrypt_token(encrypted: str, secret: str) -> str | None:
    try:
        return Fernet(_derive_demo_key(secret)).decrypt(encrypted.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return None
