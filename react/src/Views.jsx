import React from "react";
import MainPage from "./Pages/MainPage";
import Dashboard from "./Pages/Dashboard";
import { Routes, Route } from "react-router-dom";

const Views = () => {
  return (
    <Routes>
      <Route index element={<MainPage />} />
      <Route path="/dash" element={<Dashboard />} />
      <Route path="*" element={<div>404 Not Found</div>} />
    </Routes>
  );
};

export default Views;
