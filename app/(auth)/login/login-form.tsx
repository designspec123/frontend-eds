"use client"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { loginAction } from "@/lib/actions/user.actions"
import { useRouter } from "next/navigation"
import { useActionState, useEffect } from "react"
import { useFormStatus } from "react-dom"

const LoginForm = () => {
    const router=useRouter()
    const [data, action] = useActionState(loginAction, { message: "", success: false })
    const LoginButton = () => {
        const { pending } = useFormStatus()
        return <Button className="w-full" disabled={pending}>{pending ? "Login..." : "Login"}</Button>
    }
    useEffect(()=>{
        const user=localStorage.getItem('user')?JSON.parse(localStorage.getItem('user')||""):""
console.log(user,"user")
        if(user){
   window.location.href="/"
}
    },[data])
    return <form action={action}>
        <div className="space-y-6 text-white">
            <div>
                <Label htmlFor="email" className=" text-white pb-2">Email</Label>
                <Input required id="email" name="email" type="email" defaultValue={""} autoComplete="email" />
            </div>
            <div>
                <Label htmlFor="password" className=" text-white pb-2">Password</Label>
                <Input required id="password" name="password" type="password" defaultValue={""} autoComplete="email" />
            </div>
            <LoginButton />
            {data  && (<div className="text-center text-destructive">{data.message}</div>)}
        </div>
    </form>
}
export default LoginForm