export const loginAction = (prevState: unknown, formData: FormData) => {
  const email = formData.get("email");
  const password = formData.get("password");

  if (email === "admin@gmail.com" && password === "password") {
    localStorage.setItem("user", JSON.stringify({ email, password }));
    return { message: "login successfully", success: true };
  } else {
    return { message: "invalid credentials", success: false };
  }
};
