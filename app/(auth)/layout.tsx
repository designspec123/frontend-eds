import React from "react"

const Layout=({children}:Readonly<{children:React.ReactNode}>)=>{
return <div className="bg-teal-500 flex flex-col  items-center min-h-screen w-full">{children}</div>
}
export default Layout