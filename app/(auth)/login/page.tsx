import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import LoginForm from "./login-form"

const Login = () => {
    return <div
    className="h-screen "
      style={{
      background:"url('/images/eds-bg.jpg')"
    }}>
   <div className="w-full max-w-md mx-auto pt-20">
        <Card className=" bg-pink-400/80">
            <CardHeader className="space-y-4  text-white">  <CardTitle className="text-center text-2xl"></CardTitle>
                <CardDescription className="text-center  text-white">Login to your account</CardDescription></CardHeader>
<CardContent>
   <LoginForm/> 
</CardContent>

        </Card>
    </div>
    </div>
    
 
}
export default Login