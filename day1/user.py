class User:

    def __init__(self,user_id):
        self.user_id = user_id
        self._permissions = set()
    
    def add_permission(self,perm):
        self._permissions.add(perm)

    # 魔术方法实现in操作符
    def __contains__(self,perm):
        return perm in self._permissions
    
admin = User("admin")
admin.add_permission("DELETE")
print("DELETE" in admin)