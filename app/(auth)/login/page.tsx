import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import LoginForm from "./login-form"

const Login = () => {
    return <div className="w-full max-w-md mx-auto mt-20">
        <Card>
            <CardHeader className="space-y-4">  <CardTitle className="text-center text-2xl">Login</CardTitle>
                <CardDescription className="text-center">Login to your account</CardDescription></CardHeader>
<CardContent>
   <LoginForm/> 
</CardContent>

        </Card>
    </div>
}
export default Login